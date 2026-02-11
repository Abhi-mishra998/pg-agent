# Ollama Runbook AI Integration Plan

## Overview

Integrate a specialized PostgreSQL DBA Runbook AI using Ollama that provides structured output in the format:
- Diagnosis
- Evidence  
- Root Cause
- Immediate Actions
- Preventive Actions
- CLI Commands

The system will use RAG (Retrieval Augmented Generation) with knowledge base entries and deterministic pipeline results.

---

## Information Gathered

### Current Architecture
1. **LLM Layer** (`llm/llama_client.py`): Already handles Ollama API communication
2. **Pipeline** (`core/pipeline.py`): Signal → Evidence → Root Cause → Recommendation → Output
3. **Terminal Agent** (`agent/terminal_agent.py`): Interactive CLI interface
4. **Root Cause Engine** (`signals/root_cause_engine.py`): Deterministic rule-based analysis
5. **Recommender** (`recommendations/pg_recommender.py`): KB-based recommendations with DBA-safety

### Data Models Available
- `PipelineResult`: Contains signals, evidence, root causes, recommendations
- `RecommendationReport`: Contains structured recommendations
- Knowledge base entries with evidence, impact, and recommendations

### Key Files to Modify/Create
- `model/rules.md` - DBA Runbook AI system rules
- `model/examples.md` - Training examples for output format
- `agent/runbook_agent.py` - New specialized agent for runbook AI
- Integration in `agent/terminal_agent.py`

---

## Implementation Plan

### Phase 1: Create Model Files

#### 1.1 Create `model/rules.md`
```markdown
## ROLE
You are a PostgreSQL DBA Runbook AI for enterprise SRE and DBA teams.

## ALLOWED ACTIONS
- Analyze PostgreSQL metrics and statistics
- Analyze OS-level metrics (CPU, memory, disk, network)
- Identify root causes
- Recommend safe, read-only remediation steps
- Provide CLI commands as suggestions only

## STRICTLY FORBIDDEN ACTIONS
- Executing commands
- Assuming missing metrics
- Inventing configuration values
- Guessing environment details
- Recommending destructive actions without warning

## EVIDENCE REQUIREMENT
Every recommendation MUST be backed by:
- A metric
- A log observation
- Or an explicit input provided by the user

If evidence is missing:
- Clearly state what data is required
- Do NOT guess

## OUTPUT FORMAT (MANDATORY)
All responses MUST follow this exact format:

Diagnosis:
Evidence:
Root Cause:
Immediate Actions:
Preventive Actions:
CLI Commands:

No additional sections are allowed.

## INSUFFICIENT DATA HANDLING
If inputs are incomplete:
- State: "Insufficient data to conclude"
- List exactly what additional metrics are required

## TONE
- Professional
- Calm
- Precise
- No casual language
- No emojis
```

#### 1.2 Create `model/examples.md`
```markdown
## TRAINING EXAMPLES

### Example 1: Slow Query with Sequential Scan

INPUT:
- Signal: sequential_scan_detected on table "orders" (99% seq scan ratio)
- Metric: seq_scan_count=1523, table_rows=10234567
- Evidence: Query "SELECT * FROM orders WHERE customer_id = $1" taking 15s
- Root Cause: Missing index on customer_id column

OUTPUT:
Diagnosis:
PostgreSQL is performing sequential scans on the orders table for queries filtering by customer_id, causing 15+ second query times and connection pool exhaustion.

Evidence:
- Sequential scan detected on orders table (seq_scan_ratio: 0.99)
- Table contains 10,234,567 rows
- Query "SELECT * FROM orders WHERE customer_id = $1" executing in 15,234ms
- Connection pool exhausted (198/200 connections active)
- No index exists on customer_id column

Root Cause:
Missing index on customer_id column forces PostgreSQL to read all 10M+ rows from disk/memory. The query was added recently without corresponding index maintenance.

Immediate Actions:
1. Create composite index on customer_id and created_at columns
2. Run EXPLAIN ANALYZE to verify index usage
3. Verify query performance improvement

Preventive Actions:
1. Add index creation review to code deployment process
2. Monitor sequential scan percentage in pg_stat_user_tables
3. Set up alerts for sequential scans on tables >100K rows
4. Review query patterns before deploying new features

CLI Commands:
```sql
-- Create index concurrently
CREATE INDEX CONCURRENTLY idx_orders_customer_created ON orders(customer_id, created_at);

-- Verify index is used
EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM orders WHERE customer_id = 123;

-- Check sequential scan ratio
SELECT schemaname, tablename, seq_scan, idx_scan, 
       seq_scan::float / (seq_scan + idx_scan) as seq_scan_ratio
FROM pg_stat_user_tables WHERE relname = 'orders';
```

### Example 2: Blocking Transaction

INPUT:
- Signal: blocking_detected
- Metric: wait_duration=120s, blocked_pid=34521, blocking_pid=33218
- Evidence: UPDATE blocking SELECT queries

OUTPUT:
Diagnosis:
Active blocking transaction (PID 33218) holding RowExclusiveLock for 120+ seconds, blocking 5 other sessions and causing query timeouts.

Evidence:
- Blocking transaction active for 120 seconds
- 5 sessions blocked waiting for lock
- Blocking query: "BEGIN; UPDATE orders SET status = 'processing' WHERE customer_id = 12345;"
- Lock type: RowExclusiveLock
- Connection pool saturated (180/200 active)

Root Cause:
Long-running transaction holding lock while waiting for external resources (possibly application response or network). No statement_timeout configured to limit query duration.

Immediate Actions:
1. Identify blocking transaction details
2. Evaluate whether to terminate (pg_terminate_backend) or let complete
3. Cancel blocking query first (pg_cancel_backend) as less aggressive option

Preventive Actions:
1. Set statement_timeout to prevent long-running queries
2. Set idle_in_transaction_session_timeout
3. Implement connection pool query timeouts
4. Review application transaction handling
5. Ensure indexes exist for frequently updated columns

CLI Commands:
```sql
-- Identify blocking transaction
SELECT pid, usename, application_name, query, state, 
       now() - xact_start AS transaction_duration
FROM pg_stat_activity WHERE pid = 33218;

-- Check what blocker is waiting for
SELECT pid, wait_event_type, wait_event, state 
FROM pg_stat_activity WHERE pid = 33218;

-- Cancel blocking query (less disruptive)
SELECT pg_cancel_backend(33218);

-- Terminate only if necessary (rolls back transaction)
SELECT pg_terminate_backend(33218);

-- Monitor lock waits
SELECT blocked.pid AS blocked_pid, blocker.pid AS blocker_pid,
       blocked.query AS blocked_query, blocker.query AS blocker_query
FROM pg_stat_activity AS blocked
JOIN pg_stat_activity AS blocker ON blocked.pg_blocking_pids @> ARRAY[blocker.pid]
WHERE blocked.state = 'active';
```
```

---

### Phase 2: Create Runbook Agent

#### 2.1 Create `agent/runbook_agent.py`

```python
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
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from llm.llama_client import LlamaClient
from recommendations.kb_loader import KBLoader
from recommendations.kb_index import KBVectorIndex
from core.pipeline import PipelineResult


@dataclass
class RunbookOutput:
    """Structured runbook output."""
    diagnosis: str
    evidence: str
    root_cause: str
    immediate_actions: str
    preventive_actions: str
    cli_commands: str
    
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
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # LLM client
        self.llm = LlamaClient(model=model)
        
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
            with open(path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            self.logger.warning(f"File not found: {path}")
            return ""
    
    def _load_knowledge_base(self, kb_path: str) -> None:
        """Load knowledge base for RAG."""
        try:
            self.kb = self.kb_loader.load_file(kb_path)
            if self.kb and self.kb.entry_count > 0:
                self.vector_index = KBVectorIndex()
                self.vector_index.build(self.kb.entries)
                self.logger.info(f"Loaded {self.kb.entry_count} KB entries")
        except Exception as e:
            self.logger.warning(f"Could not load KB: {e}")
    
    def generate_from_pipeline_result(
        self,
        pipeline_result: PipelineResult,
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
        context = self._build_context(pipeline_result)
        
        # Get relevant KB entries
        kb_context = self._get_kb_context(pipeline_result)
        
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
    
    def _build_context(self, pipeline_result: PipelineResult) -> Dict[str, Any]:
        """Build context from pipeline results."""
        context = {
            "signals": [],
            "evidence_summary": "",
            "root_causes": [],
            "recommendations": [],
        }
        
        # Extract signals
        if pipeline_result.signal_result:
            for signal in pipeline_result.signal_result.signals:
                context["signals"].append({
                    "name": signal.name,
                    "type": signal.type,
                    "severity": signal.severity,
                    "data": signal.data,
                })
        
        # Evidence summary
        if pipeline_result.evidence_collection:
            context["evidence_summary"] = {
                "count": pipeline_result.metrics.evidence_count,
                "confidence": pipeline_result.metrics.confidence,
            }
        
        # Root causes
        if pipeline_result.root_cause_results:
            for category, result in pipeline_result.root_cause_results.items():
                if result.is_likely_cause:
                    context["root_causes"].append({
                        "category": category.value,
                        "confidence": result.confidence,
                        "factors": result.contributing_factors[:3],
                        "recommendations": result.recommendations[:3],
                    })
        
        # Recommendations
        if pipeline_result.recommendation_report:
            for rec in pipeline_result.recommendation_report.recommendations[:5]:
                context["recommendations"].append({
                    "title": rec.title,
                    "description": rec.description,
                    "actions": [a.action for a in rec.actions[:3]],
                    "risk_level": rec.risk_level,
                })
        
        return context
    
    def _get_kb_context(
        self,
        pipeline_result: PipelineResult,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get relevant KB entries for RAG."""
        if not self.vector_index or not pipeline_result.signal_result:
            return []
        
        # Build query from signal names
        symptoms = [s.name.replace("_", " ") for s in pipeline_result.signal_result.signals]
        query = " ".join(symptoms[:3])
        
        # Search KB
        results = self.vector_index.search(query, top_k=top_k)
        
        return [
            {
                "kb_id": r.entry.metadata.kb_id,
                "title": r.entry.problem_identity.issue_type,
                "description": r.entry.problem_identity.short_description,
                "severity": r.entry.metadata.severity,
                "recommendations": r.entry.get_actionable_recommendations()[:2],
            }
            for r in results
        ]
    
    def _get_kb_context_from_query(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get relevant KB entries from natural language query."""
        if not self.vector_index:
            return []
        
        results = self.vector_index.search(query, top_k=top_k)
        
        return [
            {
                "kb_id": r.entry.metadata.kb_id,
                "title": r.entry.problem_identity.issue_type,
                "description": r.entry.problem_identity.short_description,
                "severity": r.entry.metadata.severity,
                "recommendations": r.entry.get_actionable_recommendations()[:2],
            }
            for r in results
        ]
    
    def _build_prompt(
        self,
        context: Dict[str, Any],
        kb_context: List[Dict[str, Any]],
        user_question: Optional[str] = None,
    ) -> str:
        """Build the prompt for LLM."""
        # Format context as JSON for readability
        context_json = json.dumps(context, indent=2, default=str)
        
        # Format KB entries
        kb_text = ""
        for entry in kb_context:
            kb_text += f"""
### KB Entry: {entry['kb_id']}
Title: {entry['title']}
Severity: {entry['severity']}
Description: {entry['description']}
"""
        
        prompt = f"""## CONTEXT FROM ANALYSIS

### Signals Detected
{context_json}

### Relevant Knowledge Base Entries
{kb_text}

### User Question (if any)
{user_question or "Provide a comprehensive runbook response for the above analysis."}

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
            for section in sections.keys():
                if lower_line.startswith(f"{section}:"):
                    current_section = section
                    sections[section] = line_stripped[len(section) + 1:].strip()
                    break
            elif current_section and line_stripped:
                sections[current_section] += line_stripped + "\n"
        
        # Clean up
        for key in sections:
            sections[key] = sections[key].strip()
        
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


def create_runbook_agent(
    model: Optional[str] = None,
) -> RunbookAgent:
    """Factory function to create RunbookAgent."""
    return RunbookAgent(model=model)
```

---

### Phase 3: Integration with Terminal Agent

#### 3.1 Modify `agent/terminal_agent.py`

Add `runbook` command and integrate RunbookAgent:

```python
# Add to imports in terminal_agent.py
from agent.runbook_agent import RunbookAgent, RunbookOutput


# Add to TerminalAgent.__init__
self.runbook_agent = None


# Add to do_analyze method - enhance with runbook output
def do_analyze(self, args: str) -> None:
    """Enhanced analyze with runbook output."""
    # ... existing analysis code ...
    
    # Generate runbook output
    if self.runbook_agent is None:
        self.runbook_agent = RunbookAgent()
    
    if result and result.success:
        runbook_output = self.runbook_agent.generate_from_pipeline_result(
            pipeline_result=pipeline_result,  # Need to store this
            user_question=args,
        )
        self._display_runbook_output(runbook_output)


# Add new runbook command
def do_runbook(self, args: str) -> None:
    """
    runbook <question | analysis_output>
    
    Generate structured runbook response for PostgreSQL issues.
    Uses RAG with knowledge base and deterministic analysis.
    """
    if not args.strip():
        print("Usage: runbook <question | metrics_json>")
        print("\nExamples:")
        print("  runbook 'Query is slow due to sequential scan'")
        print("  runbook '{\"seq_scan_ratio\": 0.99, \"table_rows\": 1000000}'")
        return
    
    # Initialize runbook agent
    if self.runbook_agent is None:
        self.runbook_agent = RunbookAgent()
        if not self.runbook_agent.llm.is_available():
            print("⚠️ Ollama not running. Runbook AI requires Ollama.")
            print("   Run: ollama serve")
            return
    
    # Check if input is JSON (evidence)
    try:
        evidence = json.loads(args)
        runbook_output = self.runbook_agent.generate_from_query(
            query="Analyze PostgreSQL performance issue",
            evidence=evidence,
        )
    except json.JSONDecodeError:
        # Treat as natural language query
        runbook_output = self.runbook_agent.generate_from_query(
            query=args,
        )
    
    # Display output
    self._display_runbook_output(runbook_output)


def _display_runbook_output(self, output: RunbookOutput) -> None:
    """Display runbook output in terminal."""
    print("\n" + "=" * 60)
    print("📋 RUNBOOK OUTPUT")
    print("=" * 60)
    
    print(f"\n🔍 DIAGNOSIS\n{output.diagnosis}")
    print(f"\n📊 EVIDENCE\n{output.evidence}")
    print(f"\n🎯 ROOT CAUSE\n{output.root_cause}")
    print(f"\n⚡ IMMEDIATE ACTIONS\n{output.immediate_actions}")
    print(f"\n🛡️ PREVENTIVE ACTIONS\n{output.preventive_actions}")
    print(f"\n💻 CLI COMMANDS\n{output.cli_commands}")
    
    print("\n" + "=" * 60)
```

---

### Phase 4: Create Model Setup Script

#### 4.1 Create `scripts/create_runbook_model.py`

```python
#!/usr/bin/env python3
"""
Create the db-ops-runbook Ollama model from Modelfile.

This script:
1. Reads rules from model/rules.md
2. Reads examples from model/examples.md
3. Creates a Modelfile with system prompt
4. Creates the model using ollama create
"""

import os
import subprocess
import sys

def run_command(cmd: list) -> bool:
    """Run command and return success status."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {cmd[0]} succeeded")
            return True
        else:
            print(f"✗ {cmd[0]} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Command failed: {e}")
        return False

def create_modelfile() -> str:
    """Create the Modelfile."""
    # Read rules and examples
    rules_path = os.path.join(os.path.dirname(__file__), "..", "model", "rules.md")
    examples_path = os.path.join(os.path.dirname(__file__), "..", "model", "examples.md")
    
    with open(rules_path, 'r') as f:
        rules = f.read()
    
    with open(examples_path, 'r') as f:
        examples = f.read()
    
    # Create Modelfile content
    modelfile = f'''FROM llama3.1:8b

SYSTEM "
You are a PostgreSQL DBA Runbook AI for enterprise systems.

RULES:
{rules}

You must always respond in this format:
Diagnosis:
Evidence:
Root Cause:
Immediate Actions:
Preventive Actions:
CLI Commands:
"

SYSTEM "
TRAINING EXAMPLES:
{examples}
"

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
'''
    
    # Write Modelfile
    modelfile_path = "/tmp/Modelfile.db-ops-runbook"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile)
    
    return modelfile_path

def main():
    """Main entry point."""
    print("Creating db-ops-runbook Ollama model...")
    print("-" * 50)
    
    # Check if ollama is available
    if not run_command(["which", "ollama"]):
        print("ERROR: Ollama not found. Install from https://ollama.com")
        sys.exit(1)
    
    # Create Modelfile
    print("\n1. Creating Modelfile...")
    modelfile_path = create_modelfile()
    print(f"   Modelfile: {modelfile_path}")
    
    # Create model
    print("\n2. Creating model (this may take a few minutes)...")
    if not run_command(["ollama", "create", "db-ops-runbook", "-f", modelfile_path]):
        print("ERROR: Failed to create model")
        sys.exit(1)
    
    print("\n✓ db-ops-runbook model created successfully!")
    print("\nTo use:")
    print("  python main.py --runbook 'Your question here'")
    print("\nOr in terminal mode:")
    print("  runbook <question>")

if __name__ == "__main__":
    main()
```

---

### Phase 5: Test Cases

#### 5.1 Create `tests/test_runbook_agent.py`

```python
#!/usr/bin/env python3
"""
Tests for RunbookAgent with RAG functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from agent.runbook_agent import RunbookAgent, RunbookOutput


class TestRunbookAgent:
    """Test cases for RunbookAgent."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        mock = Mock()
        mock.is_available.return_value = True
        mock.generate.return_value = Mock(
            text="""Diagnosis:
Test diagnosis for slow query.

Evidence:
- Sequential scan detected on orders table
- Table has 10M rows
- Query latency 15s

Root Cause:
Missing index on customer_id column.

Immediate Actions:
1. Create index on customer_id
2. Run EXPLAIN ANALYZE

Preventive Actions:
1. Add index review to deployment
2. Monitor sequential scans

CLI Commands:
```sql
CREATE INDEX idx ON orders(customer_id);
```"""
        )
        return mock
    
    @pytest.fixture
    def runbook_agent(self, mock_llm):
        """Create RunbookAgent with mocked dependencies."""
        with patch('agent.runbook_agent.LlamaClient', return_value=mock_llm):
            agent = RunbookAgent(
                rules_path="model/rules.md",
                examples_path="model/examples.md",
                kb_path="data/kb_unified.json",
            )
            return agent
    
    def test_parse_response(self, runbook_agent):
        """Test parsing LLM response into RunbookOutput."""
        response = """Diagnosis:
Test diagnosis.

Evidence:
Test evidence.

Root Cause:
Test root cause.

Immediate Actions:
1. Action 1
2. Action 2

Preventive Actions:
1. Prevention 1

CLI Commands:
```sql
SELECT 1;
```"""
        
        output = runbook_agent._parse_response(response)
        
        assert isinstance(output, RunbookOutput)
        assert output.diagnosis == "Test diagnosis."
        assert output.evidence == "Test evidence."
        assert output.root_cause == "Test root cause."
        assert "Action 1" in output.immediate_actions
        assert "SELECT 1" in output.cli_commands
    
    def test_fallback_response(self, runbook_agent):
        """Test fallback response when LLM fails."""
        output = runbook_agent._fallback_response()
        
        assert isinstance(output, RunbookOutput)
        assert "LLM service unavailable" in output.diagnosis
    
    def test_runbook_output_to_dict(self):
        """Test RunbookOutput serialization."""
        output = RunbookOutput(
            diagnosis="Test",
            evidence="Evidence",
            root_cause="Root",
            immediate="Actions",
            preventive="Prevent",
            cli="Commands",
        )
        
        result = output.to_dict()
        
        assert result["diagnosis"] == "Test"
        assert result["evidence"] == "Evidence"


class TestRAGIntegration:
    """Test RAG integration with knowledge base."""
    
    def test_kb_context_retrieval(self):
        """Test retrieving relevant KB entries."""
        # This would test actual KB retrieval
        pass
    
    def test_hybrid_search(self):
        """Test hybrid search combining rules and semantic search."""
        # This would test the search functionality
        pass
```

---

## Dependent Files to be Created/Modified

### New Files
1. `model/` directory
2. `model/rules.md` - System rules for runbook AI
3. `model/examples.md` - Training examples
4. `agent/runbook_agent.py` - New RunbookAgent class
5. `scripts/create_runbook_model.py` - Model creation script
6. `tests/test_runbook_agent.py` - Test cases

### Modified Files
1. `agent/terminal_agent.py` - Add `runbook` command and RunbookAgent integration

---

## Followup Steps

1. **Create model directory and files**
   - Create `model/rules.md` with DBA Runbook AI rules
   - Create `model/examples.md` with training examples

2. **Create RunbookAgent**
   - Implement `agent/runbook_agent.py`
   - Add RAG integration with KB
   - Implement structured output parsing

3. **Integrate with TerminalAgent**
   - Add `runbook` command
   - Display structured output

4. **Create model setup script**
   - Create `scripts/create_runbook_model.py`
   - Document usage

5. **Test RAG functionality**
   - Create unit tests
   - Test KB retrieval
   - Test output formatting

6. **Test end-to-end**
   - Run pg-agent with runbook command
   - Verify structured output format
   - Validate CLI commands are correct

---

## Usage Examples

### After Implementation

```bash
# Create the runbook model
python scripts/create_runbook_model.py

# Run pg-agent in terminal mode
python main.py --interactive

# In terminal, use runbook command
pg-agent> runbook 'Query is slow on orders table'

# Or with JSON evidence
pg-agent> runbook '{"seq_scan_ratio": 0.99, "table_rows": 10000000}'

# Or as standalone
python main.py --runbook 'How to diagnose slow queries in PostgreSQL'
```

---

## Expected Output Format

After implementation, queries will return:

```
============================================================
📋 RUNBOOK OUTPUT
============================================================

🔍 DIAGNOSIS
PostgreSQL is performing sequential scans on the orders table for 
queries filtering by customer_id, causing 15+ second query times.

📊 EVIDENCE
- Sequential scan detected on orders table (seq_scan_ratio: 0.99)
- Table contains 10,234,567 rows
- Query "SELECT * FROM orders WHERE customer_id = $1" executing in 15,234ms
- No index exists on customer_id column

🎯 ROOT CAUSE
Missing index on customer_id column forces PostgreSQL to read all 
10M+ rows from disk/memory.

⚡ IMMEDIATE ACTIONS
1. Create composite index on customer_id and created_at columns
2. Run EXPLAIN ANALYZE to verify index usage
3. Verify query performance improvement

🛡️ PREVENTIVE ACTIONS
1. Add index creation review to code deployment process
2. Monitor sequential scan percentage in pg_stat_user_tables
3. Set up alerts for sequential scans on tables >100K rows

💻 CLI COMMANDS
```sql
-- Create index concurrently
CREATE INDEX CONCURRENTLY idx_orders_customer_created 
ON orders(customer_id, created_at);

-- Verify index is used
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM orders WHERE customer_id = 123;
```

============================================================
```

