#!/usr/bin/env python3
"""
RunbookAgent - Specialized PostgreSQL DBA Runbook AI

Provides structured runbook-style responses using RAG:
- Diagnosis
- Evidence
- Root Cause
- Immediate Actions
- Preventive Actions
- CLI Commands

Uses:
- Deterministic pipeline results (signals, evidence, root causes)
- Knowledge base entries for similar incidents
- Ollama with specialized system prompt
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm.llama_client import LlamaClient
from recommendations.kb_loader import KBLoader
from recommendations.kb_index import KBVectorIndex, SearchResult


@dataclass
class RunbookOutput:
    """Structured runbook output."""
    diagnosis: str = ""
    evidence: str = ""
    root_cause: str = ""
    immediate_actions: str = ""
    preventive_actions: str = ""
    cli_commands: str = ""
    
    def to_dict(self) -> Dict[str, str]:
        return asdict(self)
    
    def to_markdown(self) -> str:
        return f"""## Diagnosis
{self.diagnosis}

## Evidence
{self.evidence}

## Root Cause
{self.root_cause}

## Immediate Actions
{self.immediate_actions}

## Preventive Actions
{self.preventive_actions}

## CLI Commands
{self.cli_commands}
"""
    
    def is_empty(self) -> bool:
        """Check if output has any content."""
        return not any([
            self.diagnosis,
            self.evidence,
            self.root_cause,
            self.immediate_actions,
            self.preventive_actions,
            self.cli_commands,
        ])


@dataclass
class KBContextEntry:
    """Context entry from knowledge base."""
    kb_id: str
    title: str
    description: str
    severity: str
    category: str
    recommendations: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


class RunbookAgent:
    """
    Specialized agent for PostgreSQL DBA Runbook AI.
    
    Combines:
    1. Deterministic pipeline analysis
    2. Knowledge base RAG
    3. Ollama with structured output prompt
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        kb_path: str = "data/kb_unified.json",
        rules_path: str = "model/rules.md",
        examples_path: str = "model/examples.md",
    ):
        """
        Initialize the RunbookAgent.
        
        Args:
            model: Ollama model name (default: llama3.1:8b)
            kb_path: Path to knowledge base JSON file
            rules_path: Path to rules markdown file
            examples_path: Path to examples markdown file
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # LLM client with silent mode
        self.llm = LlamaClient(model=model, silent=True)
        
        # Load rules and examples
        self.rules = self._load_file(rules_path)
        self.examples = self._load_file(examples_path)
        
        # Knowledge base for RAG
        self.kb_loader = KBLoader()
        self.kb = None
        self.vector_index = None
        self._load_knowledge_base(kb_path)
        
        self.logger.info("RunbookAgent initialized")
    
    def _load_file(self, path: str) -> str:
        """Load text file."""
        try:
            p = Path(path)
            if p.exists():
                return p.read_text()
            self.logger.warning(f"File not found: {path}")
            return ""
        except Exception as e:
            self.logger.warning(f"Error loading file {path}: {e}")
            return ""
    
    def _load_knowledge_base(self, kb_path: str) -> None:
        """Load knowledge base for RAG."""
        try:
            self.kb = self.kb_loader.load_file(kb_path)
            if self.kb and self.kb.entry_count > 0:
                self.vector_index = KBVectorIndex()
                self.vector_index.build(self.kb.entries)
                self.logger.info(f"Loaded {self.kb.entry_count} KB entries")
            elif self.kb:
                self.logger.warning(f"KB has {self.kb.entry_count} entries")
        except Exception as e:
            self.logger.warning(f"Could not load KB: {e}")
    
    def is_available(self) -> bool:
        """Check if the LLM is available."""
        return self.llm.is_available()
    
    def generate_from_pipeline_result(
        self,
        pipeline_result,
        user_question: Optional[str] = None,
    ) -> RunbookOutput:
        """
        Generate runbook output from pipeline result.
        
        Args:
            pipeline_result: Result from core pipeline
            user_question: Optional user question to focus on
            
        Returns:
            RunbookOutput with structured response
        """
        # Build context from pipeline results
        context = self._build_context_from_pipeline(pipeline_result)
        
        # Get relevant KB entries
        kb_context = self._get_kb_context_from_pipeline(pipeline_result)
        
        # Build prompt
        prompt = self._build_prompt(context, kb_context, user_question)
        
        # Generate response
        response = self._generate(prompt)
        
        # Parse response
        return self._parse_response(response)
    
    def generate_from_query(
        self,
        query: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> RunbookOutput:
        """
        Generate runbook output from natural language query.
        
        Args:
            query: User's question about PostgreSQL issue
            evidence: Optional evidence/metrics to include
            
        Returns:
            RunbookOutput with structured response
        """
        context = {
            "user_query": query,
            "evidence_provided": evidence or {},
        }
        
        # Get relevant KB entries
        kb_context = self._get_kb_context_from_query(query)
        
        # Build prompt
        prompt = self._build_prompt(context, kb_context, query)
        
        # Generate response
        response = self._generate(prompt)
        
        # Parse response
        return self._parse_response(response)
    
    def generate_from_signals(
        self,
        signals: List[Dict[str, Any]],
        evidence: Optional[Dict[str, Any]] = None,
        user_question: Optional[str] = None,
    ) -> RunbookOutput:
        """
        Generate runbook output from signals dict list.
        
        Args:
            signals: List of signal dictionaries
            evidence: Optional additional evidence
            user_question: Optional user question
            
        Returns:
            RunbookOutput with structured response
        """
        context = {
            "signals": signals,
            "evidence": evidence or {},
        }
        
        # Build query from signal names
        if signals:
            symptoms = [s.get("name", "").replace("_", " ") for s in signals]
            query = " ".join(filter(None, symptoms[:3]))
        else:
            query = "PostgreSQL performance issue"
        
        kb_context = self._get_kb_context_from_query(query)
        
        # Build prompt
        prompt = self._build_prompt(context, kb_context, user_question)
        
        # Generate response
        response = self._generate(prompt)
        
        # Parse response
        return self._parse_response(response)
    
    def _build_context_from_pipeline(self, pipeline_result) -> Dict[str, Any]:
        """Build context from pipeline results."""
        context = {
            "signals": [],
            "evidence_summary": {},
            "root_causes": [],
            "recommendations": [],
            "metrics": {},
        }
        
        # Extract signals
        if hasattr(pipeline_result, 'signal_result') and pipeline_result.signal_result:
            for signal in pipeline_result.signal_result.signals:
                context["signals"].append({
                    "name": signal.name,
                    "type": signal.type,
                    "severity": signal.severity,
                    "data": signal.data,
                })
        
        # Evidence summary
        if hasattr(pipeline_result, 'evidence_collection') and pipeline_result.evidence_collection:
            context["evidence_summary"] = {
                "count": getattr(pipeline_result.metrics, 'evidence_count', 0),
                "confidence": getattr(pipeline_result.metrics, 'confidence', 0),
            }
        
        # Metrics from pipeline result
        if hasattr(pipeline_result, 'metrics') and pipeline_result.metrics:
            metrics = pipeline_result.metrics
            context["metrics"] = {
                "total_time_ms": metrics.total_time_ms,
                "signal_count": metrics.signal_count,
                "evidence_count": metrics.evidence_count,
                "confidence": metrics.confidence,
                "risk_level": metrics.risk_level,
            }
        
        # Root causes
        if hasattr(pipeline_result, 'root_cause_results') and pipeline_result.root_cause_results:
            for category, result in pipeline_result.root_cause_results.items():
                if result.is_likely_cause:
                    context["root_causes"].append({
                        "category": category.value if hasattr(category, 'value') else str(category),
                        "confidence": result.confidence,
                        "factors": result.contributing_factors[:3],
                        "recommendations": result.recommendations[:3],
                    })
        
        # Recommendations
        if hasattr(pipeline_result, 'recommendation_report') and pipeline_result.recommendation_report:
            for rec in pipeline_result.recommendation_report.recommendations[:5]:
                actions = []
                if hasattr(rec, 'actions'):
                    for a in rec.actions:
                        if hasattr(a, 'action'):
                            actions.append(a.action)
                        else:
                            actions.append(str(a))
                else:
                    actions = rec.get("actions", [])[:3]
                
                context["recommendations"].append({
                    "title": rec.title if hasattr(rec, 'title') else rec.get("title", ""),
                    "description": rec.description if hasattr(rec, 'description') else rec.get("description", ""),
                    "actions": actions,
                    "risk_level": rec.risk_level if hasattr(rec, 'risk_level') else rec.get("risk_level", ""),
                })
        
        return context
    
    def _get_kb_context_from_pipeline(
        self,
        pipeline_result,
        top_k: int = 5,
    ) -> List[KBContextEntry]:
        """Get relevant KB entries from pipeline result."""
        if not self.vector_index:
            return []
        
        # Build query from signal names
        symptoms = []
        if hasattr(pipeline_result, 'signal_result') and pipeline_result.signal_result:
            for signal in pipeline_result.signal_result.signals:
                symptoms.append(signal.name.replace("_", " "))
        
        if not symptoms:
            return []
        
        query = " ".join(symptoms[:3])
        
        return self._search_kb_context(query, top_k)
    
    def _get_kb_context_from_query(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[KBContextEntry]:
        """Get relevant KB entries from natural language query."""
        return self._search_kb_context(query, top_k)
    
    def _search_kb_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[KBContextEntry]:
        """Search KB and return context entries."""
        if not self.vector_index:
            return []
        
        try:
            results = self.vector_index.search(query, top_k=top_k)
            
            entries = []
            for r in results:
                entry = r.entry
                recs = []
                if hasattr(entry, 'recommendations') and entry.recommendations:
                    for action in entry.recommendations.immediate_actions[:2]:
                        recs.append(action.action)
                
                evidence_list = []
                if hasattr(entry, 'evidence') and entry.evidence:
                    evidence_list = [
                        f"Evidence count: {entry.confidence.evidence_count}",
                        entry.confidence.confidence_reasoning,
                    ]
                
                entries.append(KBContextEntry(
                    kb_id=entry.metadata.kb_id,
                    title=entry.problem_identity.issue_type,
                    description=entry.problem_identity.short_description,
                    severity=entry.metadata.severity,
                    category=entry.metadata.category,
                    recommendations=recs,
                    evidence=evidence_list,
                ))
            
            return entries
        except Exception as e:
            self.logger.warning(f"KB search failed: {e}")
            return []
    
    def _build_prompt(
        self,
        context: Dict[str, Any],
        kb_context: List[KBContextEntry],
        user_question: Optional[str] = None,
    ) -> str:
        """Build the prompt for LLM."""
        # Format context as JSON for readability
        context_json = json.dumps(context, indent=2, default=str)
        
        # Format KB entries
        kb_text = ""
        for entry in kb_context:
            kb_text += f"""
### KB Entry: {entry.kb_id}
Title: {entry.title}
Severity: {entry.severity}
Category: {entry.category}
Description: {entry.description}
"""
            if entry.recommendations:
                kb_text += f"Recommendations:\n"
                for rec in entry.recommendations[:2]:
                    kb_text += f"- {rec}\n"
        
        user_q = user_question or "Provide a comprehensive runbook response for the above analysis."
        
        prompt = f"""## CONTEXT FROM ANALYSIS

### Signals Detected
```
{context_json}
```

### Relevant Knowledge Base Entries
{kb_text}

### User Question
{user_q}

## INSTRUCTIONS
Based on the above analysis context and relevant KB entries, provide a structured runbook response.

Use this format exactly:

Diagnosis:
[Clear diagnosis of the issue based on signals and evidence]

Evidence:
[List the key evidence supporting this diagnosis - metrics, observations, logs]

Root Cause:
[Identify the root cause based on evidence and analysis]

Immediate Actions:
[1. First priority action]
[2. Second priority action]
[3. Third priority action]

Preventive Actions:
[1. First preventive measure]
[2. Second preventive measure]
[3. Third preventive measure]

CLI Commands:
```sql
[Relevant SQL commands to diagnose or address the issue]
[One command per line, properly formatted]
```

Remember:
- Base all conclusions on the evidence provided
- If evidence is missing, state what additional data is needed
- Provide safe, read-only diagnostic commands where possible
- Include CONCURRENTLY for index operations
"""
        return prompt
    
    def _generate(self, prompt: str) -> str:
        """Generate response from LLM."""
        system = f"""{self.rules}

TRAINING EXAMPLES:
{self.examples}
"""
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                system=system,
                options={
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_ctx": 8192,
                }
            )
            return response.text
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            return self._fallback_response()
    
    def _parse_response(self, response: str) -> RunbookOutput:
        """Parse LLM response into structured output."""
        sections = {
            "diagnosis": "",
            "evidence": "",
            "root_cause": "",
            "immediate_actions": "",
            "preventive_actions": "",
            "cli_commands": "",
        }
        
        current_section = None
        lines = response.strip().split("\n")
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check for section headers
            lower_line = line_stripped.lower()
            found_header = False
            for section in sections.keys():
                if lower_line.startswith(f"{section}:"):
                    current_section = section
                    sections[section] = line_stripped[len(section) + 1:].strip()
                    found_header = True
                    break
            
            # If not a header and we're in a section, add to it
            if not found_header and current_section and line_stripped:
                sections[current_section] += line_stripped + "\n"
        
        # Clean up - remove trailing newlines and normalize
        for key in sections:
            sections[key] = sections[key].strip()
            # Ensure lists are properly formatted
            if key in ["immediate_actions", "preventive_actions"]:
                if sections[key] and not sections[key].startswith("1."):
                    # Add numbering if missing
                    lines = [l.strip() for l in sections[key].split("\n") if l.strip()]
                    sections[key] = "\n".join([f"{i+1}. {l}" for i, l in enumerate(lines)])
        
        return RunbookOutput(**sections)
    
    def _fallback_response(self) -> str:
        """Generate fallback response when LLM fails."""
        return """Diagnosis:
Unable to generate detailed diagnosis due to LLM availability issues.

Evidence:
Pipeline analysis completed but LLM generation failed.

Root Cause:
Unknown - LLM service unavailable.

Immediate Actions:
1. Check that Ollama server is running
2. Verify model 'llama3.1:8b' is available
3. Retry the query

Preventive Actions:
1. Monitor LLM service availability
2. Set up alerting for Ollama server

CLI Commands:
```sql
-- Check Ollama status
ollama list

-- Test model availability
ollama run llama3.1:8b "test"
```
"""
    
    def analyze_similar_incidents(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Find similar incidents in the knowledge base."""
        if not self.vector_index:
            return []
        
        return self.vector_index.search(query, top_k=top_k)


def create_runbook_agent(
    model: Optional[str] = None,
) -> RunbookAgent:
    """Factory function to create RunbookAgent."""
    return RunbookAgent(model=model)


# =====================================================================
# EXAMPLE USAGE
# =====================================================================

if __name__ == "__main__":
    # Example: Using the RunbookAgent
    import logging
    logging.basicConfig(level=logging.INFO)
    
    agent = RunbookAgent()
    
    if not agent.is_available():
        print("Ollama not available. Run: ollama serve")
        exit(1)
    
    print("RunbookAgent initialized successfully!")
    
    # Example: Generate from query
    print("\n--- Example: Generate from query ---")
    result = agent.generate_from_query(
        "Sequential scan on large table causing slow queries"
    )
    
    print(f"\nDiagnosis: {result.diagnosis[:100]}...")
    print(f"Root Cause: {result.root_cause[:100]}...")
    print(f"CLI Commands available: {'Yes' if result.cli_commands else 'No'}")

