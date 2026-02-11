#!/usr/bin/env python3
"""
pg_recommender.py

PostgreSQL DBA-Safe Recommendation Engine with Knowledge Base

Purpose:
- Convert signals + evidence into actionable recommendations
- Use deterministic expert rules (NO hallucination)
- Use KB for similar past incidents and best practices
- Provide explainable, client-ready, DBA-safe guidance
- Hybrid retrieval: rule-based + similarity-based

DBA-Safe Features:
- Each recommendation includes risk level (low/medium/high)
- Online vs Offline operation specified
- Approval requirements marked
- Rollback notes provided
- Destructive SQL never generated without approval flag
- CONCURRENTLY preferred for index operations
- No VACUUM FULL without explicit justification
- Parameter changes include restart warnings

Inputs:
- SignalResult
- EvidenceCollection
- KB entries from knowledge base

Output:
- Structured recommendations with KB references and DBA-safety info
"""

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from signals.signal_engine import SignalResult, Signal
from signals.evidence_builder import EvidenceCollection

from .kb_schema import KBEntry, KBVersion
from .kb_schema import Severity, Category, Action, Recommendations
from .kb_loader import KBLoader
from .kb_index import KBVectorIndex, KBRuleMatcher, SearchResult


# -------------------------------------------------------------------
# Data Models
# -------------------------------------------------------------------

@dataclass
class DBAction:
    """
    A single DBA-safe action with full safety information.
    
    This extends the base Action with display-friendly fields
    for the recommendation output.
    """
    action: str
    sql: Optional[str] = None
    risk_level: str = "low"  # low/medium/high
    is_online: bool = True  # Can run while database is active
    requires_approval: bool = False  # Requires DBA approval
    rollback_notes: str = ""  # How to rollback this action
    priority: str = ""  # immediate/long-term
    
    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "sql": self.sql,
            "risk_level": self.risk_level,
            "is_online": self.is_online,
            "requires_approval": self.requires_approval,
            "rollback_notes": self.rollback_notes,
            "priority": self.priority,
        }


@dataclass
class Recommendation:
    """
    A DBA-safe recommendation with full safety information.
    
    Each recommendation includes:
    - Action description
    - SQL (if applicable)
    - Risk level (low/medium/high)
    - Online vs Offline
    - Requires approval (yes/no)
    - Rollback notes
    """
    category: str
    severity: str
    title: str
    description: str
    actions: List[DBAction]
    confidence: float
    # DBA-safe fields
    risk_level: str = "low"  # low/medium/high
    requires_approval: bool = False
    references: List[str] = field(default_factory=list)
    kb_entry_id: Optional[str] = None
    match_type: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    root_cause: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "category": self.category,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "actions": [a.to_dict() for a in self.actions],
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "requires_approval": self.requires_approval,
            "references": self.references,
            "kb_entry_id": self.kb_entry_id,
            "match_type": self.match_type,
            "evidence": self.evidence,
            "root_cause": self.root_cause,
        }


@dataclass
class RecommendationReport:
    """A complete DBA-safe recommendation report."""
    recommendations: List[Recommendation]
    risk_level: str
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    kb_references: List[Dict[str, Any]] = field(default_factory=list)
    actions_requiring_approval: List[Dict[str, Any]] = field(default_factory=list)
    offline_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "recommendations": [r.to_dict() for r in self.recommendations],
            "risk_level": self.risk_level,
            "summary": self.summary,
            "metadata": self.metadata,
            "kb_references": self.kb_references,
            "actions_requiring_approval": self.actions_requiring_approval,
            "offline_actions": self.offline_actions,
        }


# -------------------------------------------------------------------
# DBA-Safe SQL Rules
# -------------------------------------------------------------------

# Operations that always require approval
DESTRUCTIVE_OPERATIONS = {
    "DROP": "DROP operations permanently remove database objects",
    "TRUNCATE": "TRUNCATE removes all rows from a table",
    "VACUUM_FULL": "VACUUM FULL requires exclusive lock and can take significant time",
    "REINDEX": "Standard REINDEX locks writes (use CONCURRENTLY)",
    "ALTER_TABLE_REWRITE": "ALTER TABLE operations that rewrite the table",
    "pg_terminate_backend": "Terminating backend can interrupt active transactions",
    "pg_cancel_backend": "Canceling backend can interrupt active queries",
}

# Operations that are safe online
SAFE_ONLINE_OPERATIONS = {
    "SELECT",
    "EXPLAIN",
    "EXPLAIN_ANALYZE",
    "ANALYZE",
    "CREATE_INDEX_CONCURRENTLY",
    "CREATE_INDEX",
    "REINDEX_CONCURRENTLY",
}

# VACUUM FULL justification patterns
VACUUM_FULL_JUSTIFICATIONS = [
    "severe index bloat",
    "table bloat over 50%",
    "corrupted data",
    "disk space reclamation critical",
]


# -------------------------------------------------------------------
# Root Cause to Actions Mapping
# -------------------------------------------------------------------

ROOT_CAUSE_TO_ACTIONS = {
    "INDEX_ISSUES": {
        "missing_index": {
            "action": "Create index on frequently queried columns",
            "sql_template": "CREATE INDEX CONCURRENTLY idx_{table}_{columns} ON {table}({columns});",
            "risk": "medium",
            "is_online": True,
            "requires_approval": False,
            "rollback_notes": "DROP INDEX CONCURRENTLY idx_{table}_{columns};",
            "priority": "immediate",
        },
        "sequential_scan": {
            "action": "Add index to eliminate sequential scan",
            "sql_template": "CREATE INDEX CONCURRENTLY idx_{table}_{columns} ON {table}({columns});",
            "risk": "medium",
            "is_online": True,
            "requires_approval": False,
            "rollback_notes": "DROP INDEX CONCURRENTLY idx_{table}_{columns};",
            "priority": "immediate",
        },
        "high_index_bloat": {
            "action": "Reindex to remove bloat",
            "sql_template": "REINDEX CONCURRENTLY {index_name};",
            "risk": "medium",
            "is_online": True,
            "requires_approval": False,
            "rollback_notes": "Not needed - REINDEX is recovery operation",
            "priority": "long-term",
        },
        "unused_index": {
            "action": "Consider dropping unused index",
            "sql_template": "DROP INDEX CONCURRENTLY {index_name};",
            "risk": "high",
            "is_online": True,
            "requires_approval": True,
            "rollback_notes": "Recreate index: CREATE INDEX CONCURRENTLY {index_definition};",
            "priority": "long-term",
        },
    },
    "STATISTICS_MAINTENANCE": {
        "stale_statistics": {
            "action": "Update statistics with ANALYZE",
            "sql_template": "ANALYZE {table_name};",
            "risk": "low",
            "is_online": True,
            "requires_approval": False,
            "rollback_notes": "Not needed - ANALYZE only collects statistics",
            "priority": "immediate",
        },
        "high_dead_tuples": {
            "action": "Run VACUUM to reclaim space",
            "sql_template": "VACUUM (VERBOSE) {table_name};",
            "risk": "low",
            "is_online": True,
            "requires_approval": False,
            "rollback_notes": "Not needed - VACUUM only reclaims space",
            "priority": "immediate",
        },
        "row_estimation_error": {
            "action": "Run ANALYZE to improve row estimates",
            "sql_template": "ANALYZE {table_name};",
            "risk": "low",
            "is_online": True,
            "requires_approval": False,
            "rollback_notes": "Not needed - ANALYZE only collects statistics",
            "priority": "immediate",
        },
    },
    "BLOCKING_LOCKING": {
        "blocking_detected": {
            "action": "Identify and handle blocking transaction",
            "sql_template": "SELECT pg_blocking_pids({blocked_pid});",
            "risk": "high",
            "is_online": True,
            "requires_approval": True,
            "rollback_notes": "Let transaction complete naturally if possible",
            "priority": "immediate",
        },
        "lock_wait": {
            "action": "Check lock timeout settings",
            "sql_template": "SHOW lock_timeout;",
            "risk": "low",
            "is_online": True,
            "requires_approval": False,
            "rollback_notes": "Not applicable - read-only operation",
            "priority": "immediate",
        },
        "long_running_transaction": {
            "action": "Review long-running transactions",
            "sql_template": "SELECT pid, now() - xact_start AS duration, query FROM pg_stat_activity WHERE state != 'idle' ORDER BY xact_start;",
            "risk": "low",
            "is_online": True,
            "requires_approval": False,
            "rollback_notes": "Not applicable - read-only operation",
            "priority": "immediate",
        },
    },
    "CONFIGURATION": {
        "low_shared_buffers": {
            "action": "Increase shared_buffers (requires restart)",
            "sql_template": "ALTER SYSTEM SET shared_buffers = '{value}MB';",
            "risk": "high",
            "is_online": False,
            "requires_approval": True,
            "rollback_notes": "ALTER SYSTEM SET shared_buffers = '{old_value}'; Requires PostgreSQL restart",
            "priority": "long-term",
        },
        "insufficient_work_mem": {
            "action": "Increase work_mem (per operation)",
            "sql_template": "SET work_mem = '{value}MB';",
            "risk": "medium",
            "is_online": True,
            "requires_approval": False,
            "rollback_notes": "Reset with: RESET work_mem;",
            "priority": "immediate",
        },
        "plan_regression": {
            "action": "Review query plan changes",
            "sql_template": "EXPLAIN (ANALYZE, BUFFERS) {query};",
            "risk": "low",
            "is_online": True,
            "requires_approval": False,
            "rollback_notes": "Not applicable - read-only operation",
            "priority": "immediate",
        },
    },
    "CAPACITY_HARDWARE": {
        "disk_io_bottleneck": {
            "action": "Monitor I/O metrics and consider storage upgrade",
            "sql_template": "SELECT * FROM pg_stat_disk_io WHERE datname = current_database();",
            "risk": "low",
            "is_online": True,
            "requires_approval": False,
            "rollback_notes": "Not applicable - read-only operation",
            "priority": "long-term",
        },
        "memory_pressure": {
            "action": "Check system memory usage",
            "sql_template": "SELECT * FROM pg_os_info;",
            "risk": "low",
            "is_online": True,
            "requires_approval": False,
            "rollback_notes": "Not applicable - read-only operation",
            "priority": "long-term",
        },
    },
    "BACKGROUND_JOBS": {
        "autovacuum_interference": {
            "action": "Review autovacuum settings",
            "sql_template": "SHOW autovacuum_vacuum_cost_delay;",
            "risk": "low",
            "is_online": True,
            "requires_approval": False,
            "rollback_notes": "Reset with: ALTER SYSTEM RESET autovacuum_vacuum_cost_delay;",
            "priority": "long-term",
        },
    },
    "APPLICATION_BEHAVIOR": {
        "n_plus_one_queries": {
            "action": "Batch queries using IN clauses or JOINs",
            "sql_template": "-- Rewrite multiple single queries to single query with IN clause",
            "risk": "low",
            "is_online": True,
            "requires_approval": False,
            "rollback_notes": "Not applicable - application change",
            "priority": "long-term",
        },
        "unbounded_result_set": {
            "action": "Add LIMIT clause or implement pagination",
            "sql_template": "-- Add LIMIT and OFFSET for pagination",
            "risk": "low",
            "is_online": True,
            "requires_approval": False,
            "rollback_notes": "Not applicable - application change",
            "priority": "immediate",
        },
    },
    "DEPLOYMENT_SCHEMA": {
        "recent_deployment": {
            "action": "Review recent deployment changes",
            "sql_template": "-- Check pg_stat_statements for new slow queries",
            "risk": "low",
            "is_online": True,
            "requires_approval": False,
            "rollback_notes": "Not applicable - read-only operation",
            "priority": "immediate",
        },
        "schema_change_impact": {
            "action": "Review recent DDL changes",
            "sql_template": "SELECT * FROM pg_stat_activity WHERE state != 'idle' AND query ILIKE '%ALTER%';",
            "risk": "low",
            "is_online": True,
            "requires_approval": False,
            "rollback_notes": "Not applicable - read-only operation",
            "priority": "immediate",
        },
    },
}


# -------------------------------------------------------------------
# KB-Enhanced Recommender Engine
# -------------------------------------------------------------------

class PgRecommender:
    """
    PostgreSQL expert rule-based recommender with KB integration.
    
    This class combines:
    1. Deterministic expert rules (no hallucination)
    2. Knowledge base for similar incidents
    3. Hybrid retrieval (rule + similarity)
    4. Explainable recommendations
    """
    
    def __init__(self, kb_path: str = "data/kb_unified.json"):
        """
        Initialize the recommender.
        
        Args:
            kb_path: Path to KB JSON file
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load knowledge base
        self.kb_loader = KBLoader()
        self.kb: Optional[KBVersion] = None
        self.vector_index: Optional[KBVectorIndex] = None
        self.rule_matcher = KBRuleMatcher()
        
        # Load KB
        self._load_knowledge_base(kb_path)
        
        # Initialize rule matcher
        self._setup_rule_matcher()
    
    def _load_knowledge_base(self, kb_path: str) -> None:
        """Load the knowledge base."""
        try:
            self.kb = self.kb_loader.load_file(kb_path)
            self.logger.info(f"Loaded {self.kb.entry_count} KB entries from {kb_path}")
            
            # Build vector index for similarity search
            if self.kb.entry_count > 0:
                self.vector_index = KBVectorIndex()
                self.vector_index.build(self.kb.entries)
                self.logger.info("Vector index built successfully")
        except Exception as e:
            self.logger.warning(f"Could not load knowledge base: {e}")
            self.kb = None
    
    def _setup_rule_matcher(self) -> None:
        """Setup rule matching patterns."""
        # Add symptom-based rules
        self.rule_matcher.add_rule(
            name="slow_query",
            condition="symptom",
            category="query_performance",
            priority=10
        )
        self.rule_matcher.add_rule(
            name="high_latency",
            condition="symptom",
            category="query_performance",
            priority=10
        )
        self.rule_matcher.add_rule(
            name="deadlock",
            condition="symptom",
            category="locking",
            priority=9
        )
        self.rule_matcher.add_rule(
            name="blocking",
            condition="symptom",
            category="locking",
            priority=9
        )
        self.rule_matcher.add_rule(
            name="vacuum_needed",
            condition="symptom",
            category="maintenance",
            priority=8
        )
        self.rule_matcher.add_rule(
            name="stale_stats",
            condition="symptom",
            category="maintenance",
            priority=8
        )
        self.rule_matcher.add_rule(
            name="missing_index",
            condition="symptom",
            category="index_health",
            priority=7
        )
        self.rule_matcher.add_rule(
            name="sequential_scan",
            condition="symptom",
            category="query_performance",
            priority=7
        )
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        signal_result: SignalResult,
        evidence: EvidenceCollection,
    ) -> RecommendationReport:
        """
        Generate recommendations based on signals and evidence.
        
        Uses hybrid approach:
        1. Rule-based matching for known patterns
        2. Similarity-based retrieval for similar incidents
        3. KB augmentation for comprehensive recommendations
        """
        recommendations: List[Recommendation] = []
        kb_references: List[Dict[str, Any]] = []
        
        # Extract symptoms and causes from signals
        symptoms = self._extract_symptoms(signal_result)
        causes = self._extract_causes(signal_result)
        
        # Get KB entries using rule matching
        if self.kb and self.kb.entry_count > 0:
            rule_results = self.rule_matcher.match(
                kb=self.kb,
                symptoms=symptoms,
                causes=causes,
                top_k=10
            )
            
            # Get KB entries using semantic search
            if self.vector_index and symptoms:
                query_text = " ".join(symptoms[:3])
                semantic_results = self.vector_index.search(query_text, top_k=5)
                
                # Merge results
                all_kb_results = self._merge_kb_results(rule_results, semantic_results)
                
                # Extract recommendations from KB
                for result in all_kb_results:
                    kb_rec = self._kb_entry_to_recommendation(result)
                    if kb_rec:
                        recommendations.append(kb_rec)
                        kb_references.append(result.to_dict())
        
        # Generate rule-based recommendations
        for signal in signal_result.signals:
            rule_recs = self._generate_rule_recommendations(signal)
            recommendations.extend(rule_recs)
        
        # Augment with KB context
        recommendations = self._augment_with_kb(recommendations, kb_references)
        
        # Calculate risk level
        risk_level = self._calculate_risk(signal_result.signals, recommendations)
        
        # Build summary
        summary = self._build_summary(recommendations, risk_level, len(kb_references))
        
        return RecommendationReport(
            recommendations=recommendations,
            risk_level=risk_level,
            summary=summary,
            metadata={
                "signal_count": len(signal_result.signals),
                "evidence_count": len(evidence.evidence),
                "overall_confidence": evidence.overall_confidence,
                "kb_entries_used": len(kb_references),
            },
            kb_references=kb_references,
        )
    
    def recommend_from_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RecommendationReport:
        """
        Generate recommendations from a natural language query.
        
        Args:
            query: Natural language query about PostgreSQL issue
            context: Optional context (tables, error messages, etc.)
            
        Returns:
            RecommendationReport with matching KB entries
        """
        recommendations: List[Recommendation] = []
        kb_references: List[Dict[str, Any]] = []
        
        if self.vector_index:
            # Semantic search
            semantic_results = self.vector_index.search(query, top_k=10)
            
            # Keyword search
            keywords = self._extract_keywords(query)
            keyword_results = self.vector_index.keyword_search(keywords, top_k=5)
            
            # Hybrid search
            hybrid_results = self.vector_index.hybrid_search(
                query=query,
                keywords=keywords,
                top_k=10
            )
            
            # Convert to recommendations
            for result in hybrid_results:
                kb_rec = self._kb_entry_to_recommendation(result)
                if kb_rec:
                    kb_rec.match_type = result.match_type
                    recommendations.append(kb_rec)
                    kb_references.append(result.to_dict())
        
        # Add context-based recommendations
        if context:
            context_recs = self._generate_context_recommendations(context)
            recommendations.extend(context_recs)
        
        risk_level = "MEDIUM" if recommendations else "LOW"
        summary = self._build_summary(recommendations, risk_level, len(kb_references))
        
        return RecommendationReport(
            recommendations=recommendations,
            risk_level=risk_level,
            summary=summary,
            metadata={
                "query": query,
                "kb_entries_used": len(kb_references),
                "context_provided": context is not None,
            },
            kb_references=kb_references,
        )
    
    def get_kb_entry_by_id(self, kb_id: str) -> Optional[KBEntry]:
        """Get a specific KB entry by ID."""
        if not self.kb:
            return None
        for entry in self.kb.entries:
            if entry.metadata.kb_id == kb_id:
                return entry
        return None
    
    def get_similar_entries(
        self,
        entry_id: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Get entries similar to a given KB entry."""
        if not self.vector_index:
            return []
        
        entry = self.get_kb_entry_by_id(entry_id)
        if not entry:
            return []
        
        return self.vector_index.get_similar_entries(entry, top_k=top_k)
    
    def get_entries_by_category(self, category: str) -> List[KBEntry]:
        """Get all KB entries in a category."""
        if not self.kb:
            return []
        return self.kb_loader.get_entries_by_category(self.kb, category)
    
    def get_entries_by_severity(self, severity: str) -> List[KBEntry]:
        """Get all KB entries with a severity level."""
        if not self.kb:
            return []
        return self.kb_loader.get_entries_by_severity(self.kb, severity)
    
    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------
    
    def _extract_symptoms(self, signal_result: SignalResult) -> List[str]:
        """Extract symptoms from signals."""
        symptoms = []
        for signal in signal_result.signals:
            if signal.name:
                symptoms.append(signal.name.replace("_", " "))
            if signal.data:
                if "latency" in str(signal.data).lower():
                    symptoms.append("high latency")
                if "deadlock" in str(signal.data).lower():
                    symptoms.append("deadlock")
                if "blocking" in str(signal.data).lower():
                    symptoms.append("blocking")
                if "sequential" in str(signal.data).lower():
                    symptoms.append("sequential scan")
                if "vacuum" in str(signal.data).lower():
                    symptoms.append("vacuum needed")
        return list(set(symptoms))
    
    def _extract_causes(self, signal_result: SignalResult) -> List[str]:
        """Extract causes from signals."""
        causes = []
        for signal in signal_result.signals:
            if "statistics" in str(signal.data).lower():
                causes.append("stale statistics")
            if "index" in str(signal.data).lower():
                causes.append("missing index")
            if "work_mem" in str(signal.data).lower():
                causes.append("insufficient memory")
            if "lock" in str(signal.data).lower():
                causes.append("lock contention")
        return list(set(causes))
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from natural language query."""
        keywords = []
        query_lower = query.lower()
        
        # Common PostgreSQL issue keywords
        keyword_map = {
            "slow": ["slow", "performance", "latency"],
            "lock": ["lock", "deadlock", "blocking", "wait"],
            "index": ["index", "scan", "bloat"],
            "vacuum": ["vacuum", "analyze", "dead tuple", "bloat"],
            "memory": ["memory", "work_mem", "shared_buffers"],
            "query": ["query", "execution", "plan"],
        }
        
        for key, variants in keyword_map.items():
            for variant in variants:
                if variant in query_lower:
                    keywords.append(key)
                    break
        
        return list(set(keywords))
    
    def _merge_kb_results(
        self,
        rule_results: List[SearchResult],
        semantic_results: List[SearchResult]
    ) -> List[SearchResult]:
        """Merge rule-based and semantic search results."""
        # Use dictionary to avoid duplicates
        seen = {}
        
        for result in rule_results:
            key = result.entry.metadata.kb_id
            if key not in seen or seen[key].score < result.score:
                seen[key] = result
        
        for result in semantic_results:
            key = result.entry.metadata.kb_id
            if key not in seen or seen[key].score < result.score:
                seen[key] = result
        
        # Sort by score
        sorted_results = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return sorted_results
    
    def _kb_entry_to_recommendation(
        self,
        result: SearchResult
    ) -> Optional[Recommendation]:
        """Convert KB entry to recommendation."""
        entry = result.entry
        
        if not entry.get_actionable_recommendations():
            return None
        
        return Recommendation(
            category=entry.metadata.category,
            severity=entry.metadata.severity,
            title=f"[KB] {entry.problem_identity.issue_type}",
            description=entry.problem_identity.short_description,
            actions=entry.get_actionable_recommendations()[:5],
            confidence=entry.confidence.confidence_score,
            references=entry.get_sql_examples()[:2] + entry.get_config_examples()[:2],
            kb_entry_id=entry.metadata.kb_id,
            match_type=result.match_type,
            evidence=[
                f"Evidence count: {entry.confidence.evidence_count}",
                entry.confidence.confidence_reasoning,
            ],
        )
    
    def _generate_rule_recommendations(
        self,
        signal: Signal
    ) -> List[Recommendation]:
        """Generate recommendations based on rules."""
        recs: List[Recommendation] = []
        
        # Route to appropriate rule set
        rule_map = {
            "sql": self._sql_rules,
            "pgbench": self._pgbench_rules,
            "text": self._text_rules,
            "json_context": self._json_context_rules,
            "incident": self._incident_rules,
            "index_health": self._index_health_rules,
            "query_metrics": self._query_metrics_rules,
            "table_stats": self._table_stats_rules,
        }
        
        rule_func = rule_map.get(signal.type, self._default_rules)
        recs = rule_func(signal)
        
        return recs
    
    def _generate_context_recommendations(
        self,
        context: Dict[str, Any]
    ) -> List[Recommendation]:
        """Generate recommendations from context."""
        recs = []
        
        # Check for specific context patterns
        if "table" in context:
            recs.append(Recommendation(
                category="General",
                severity="low",
                title=f"Context: {context['table']}",
                description=f"Table {context['table']} mentioned in context",
                actions=[
                    f"Check table statistics: SELECT * FROM pg_stat_user_tables WHERE relname = '{context['table']}'",
                    f"Check index usage: SELECT * FROM pg_stat_user_indexes WHERE relname = '{context['table']}'",
                ],
                confidence=0.7,
            ))
        
        return recs
    
    def _augment_with_kb(
        self,
        recommendations: List[Recommendation],
        kb_references: List[Dict[str, Any]]
    ) -> List[Recommendation]:
        """Augment recommendations with KB context."""
        # Add KB reference info to recommendations
        for i, rec in enumerate(recommendations):
            if i < len(kb_references):
                if rec.kb_entry_id is None:
                    rec.kb_entry_id = kb_references[i].get("kb_id")
                if rec.match_type is None:
                    rec.match_type = kb_references[i].get("match_type", "rule")
        
        return recommendations
    
    def _calculate_risk(
        self,
        signals: List[Signal],
        recommendations: List[Recommendation]
    ) -> str:
        """Calculate overall risk level."""
        # Check for critical severity
        if any(s.severity == "critical" for s in signals):
            return "CRITICAL"
        if any(r.severity == "critical" for r in recommendations):
            return "CRITICAL"
        
        # Check for high severity
        if any(s.severity == "high" for s in signals):
            return "HIGH"
        if any(r.severity == "high" for r in recommendations):
            return "HIGH"
        
        # Check for medium severity
        if any(s.severity == "medium" for s in signals):
            return "MEDIUM"
        if any(r.severity == "medium" for r in recommendations):
            return "MEDIUM"
        
        return "LOW"
    
    def _build_summary(
        self,
        recommendations: List[Recommendation],
        risk_level: str,
        kb_count: int
    ) -> str:
        """Build recommendation summary."""
        if not recommendations:
            return "No issues detected. System appears healthy."
        
        kb_note = f" ({kb_count} KB entries used)" if kb_count > 0 else ""
        
        return (
            f"Risk Level: {risk_level}. "
            f"{len(recommendations)} recommendation(s) generated{kb_note}. "
            "Immediate attention is advised for high or critical items."
        )
    
    # ------------------------------------------------------------------
    # Rule Sets (from original recommender, adapted)
    # ------------------------------------------------------------------
    
    def _default_rules(self, signal: Signal) -> List[Recommendation]:
        """Default rules for unknown signal types."""
        return [
            Recommendation(
                category="General",
                severity="medium",
                title=f"Signal: {signal.name}",
                description=str(signal.data),
                actions=[
                    "Review signal details",
                    "Check PostgreSQL logs",
                    "Verify system health",
                ],
                confidence=0.7,
            )
        ]
    
    def _sql_rules(self, signal: Signal) -> List[Recommendation]:
        """SQL-related rules."""
        recs = []
        if signal.name == "destructive_sql":
            recs.append(Recommendation(
                category="Data Safety",
                severity="critical",
                title="Destructive SQL without WHERE clause",
                description="A DELETE without WHERE was detected. This removes all rows!",
                actions=[
                    "Immediately stop the transaction if still running",
                    "Restore data from the latest backup if data loss occurred",
                    "Add WHERE clauses to DELETE statements",
                    "Use transactions with ROLLBACK during testing",
                    "Restrict DELETE permissions in production roles",
                ],
                confidence=0.95,
                references=["https://www.postgresql.org/docs/current/sql-delete.html"],
            ))
        return recs
    
    def _pgbench_rules(self, signal: Signal) -> List[Recommendation]:
        """pgbench-related rules."""
        recs = []
        if signal.name == "low_tps":
            recs.append(Recommendation(
                category="Performance",
                severity="high",
                title="Low TPS detected",
                description="Transactions per second are below expected thresholds.",
                actions=[
                    "Check for missing indexes on frequently accessed tables",
                    "Analyze slow queries using pg_stat_statements",
                    "Increase shared_buffers if memory allows",
                    "Ensure autovacuum is running properly",
                ],
                confidence=0.9,
            ))
        return recs
    
    def _text_rules(self, signal: Signal) -> List[Recommendation]:
        """Text/log-related rules."""
        return [
            Recommendation(
                category="Diagnostics",
                severity="medium",
                title="Error keyword detected in logs",
                description=f"Error pattern detected: {signal.data}",
                actions=[
                    "Review PostgreSQL logs around the timestamp",
                    "Enable log_min_error_statement",
                    "Correlate errors with recent deployments",
                ],
                confidence=0.6,
            )
        ]
    
    def _json_context_rules(self, signal: Signal) -> List[Recommendation]:
        """JSON context rules for performance patterns."""
        recs = []
        
        if signal.name == "missing_index_large_table":
            table = signal.data.get("table", "unknown")
            recs.append(Recommendation(
                category="Performance",
                severity="high",
                title=f"Missing Index on Table '{table}'",
                description=f"Table '{table}' has no indexes for large queries.",
                actions=[
                    f"CREATE INDEX idx_{table}_on_column ON {table}(column_name)",
                    "Identify frequently queried columns using pg_stat_statements",
                    "Consider composite indexes for multi-column WHERE clauses",
                ],
                confidence=0.92,
            ))
        
        if signal.name == "sequential_scan_detected":
            table = signal.data.get("table", "unknown")
            recs.append(Recommendation(
                category="Performance",
                severity="high",
                title=f"Sequential Scan Detected on '{table}'",
                description="Sequential scan on large table indicates missing index.",
                actions=[
                    f"CREATE INDEX on columns used in WHERE clauses for {table}",
                    "Use SET enable_seqscan = off to test index performance",
                    "Run ANALYZE after index creation",
                ],
                confidence=0.90,
            ))
        
        return recs
    
    def _incident_rules(self, signal: Signal) -> List[Recommendation]:
        """Incident-related rules."""
        recs = []
        
        if signal.name == "active_incident":
            incident_type = signal.data.get("incident_type", "unknown")
            severity = signal.data.get("severity", "unknown")
            recs.append(Recommendation(
                category="Incident Response",
                severity=signal.severity,
                title=f"Active {severity} Incident: {incident_type}",
                description=f"Incident is active with severity {severity}.",
                actions=[
                    "Acknowledge the incident in your monitoring system",
                    "Check PostgreSQL logs for error details",
                    "Identify root cause using pg_stat_statements",
                    "Consider escalating to senior DBA if unresolved",
                ],
                confidence=0.95,
            ))
        
        if signal.name == "query_performance_alert":
            deviation = signal.data.get("deviation_factor", 0)
            recs.append(Recommendation(
                category="Performance",
                severity="critical",
                title=f"Query Performance Alert: {deviation}x Deviation",
                description=f"Query executing at {deviation}x expected time.",
                actions=[
                    "Run EXPLAIN ANALYZE on the affected query",
                    "Check if statistics are stale - run ANALYZE",
                    "Verify no recent schema changes caused plan regression",
                    "Check for lock contention blocking the query",
                ],
                confidence=0.95,
            ))
        
        if signal.name == "blocking_detected":
            recs.append(Recommendation(
                category="Concurrency",
                severity="high",
                title="Blocking Transaction Detected",
                description="Blocking transaction is causing performance issues.",
                actions=[
                    "Identify blocking session using pg_blocking_pids()",
                    "Check what query the blocking session is running",
                    "Evaluate whether to terminate blocking session",
                    "Consider setting statement_timeout to prevent long locks",
                ],
                confidence=0.95,
            ))
        
        return recs
    
    def _index_health_rules(self, signal: Signal) -> List[Recommendation]:
        """Index health rules."""
        recs = []
        
        if signal.name == "high_index_bloat":
            index_name = signal.data.get("index_name", "unknown")
            bloat_percent = signal.data.get("bloat_percent", 0)
            recs.append(Recommendation(
                category="Index Health",
                severity="medium",
                title=f"High Index Bloat: {index_name}",
                description=f"Index has {bloat_percent}% bloat.",
                actions=[
                    f"SCHEDULE REINDEX CONCURRENTLY {index_name}",
                    "Check if frequent updates cause bloat",
                    "Consider increasing autovacuum_vacuum_cost_delay",
                ],
                confidence=0.88,
            ))
        
        return recs
    
    def _query_metrics_rules(self, signal: Signal) -> List[Recommendation]:
        """Query metrics rules."""
        recs = []
        
        if signal.name == "high_query_latency_deviation":
            deviation = signal.data.get("deviation_factor", 0)
            recs.append(Recommendation(
                category="Performance",
                severity="critical",
                title=f"Query Latency {deviation}x Above Baseline",
                description="Severe performance regression detected.",
                actions=[
                    "Run EXPLAIN ANALYZE to identify the issue",
                    "Check if statistics are stale - run ANALYZE",
                    "Verify no recent schema changes affected query plans",
                    "Check for lock contention",
                ],
                confidence=0.95,
            ))
        
        if signal.name == "high_temp_file_usage":
            temp_blocks = signal.data.get("temp_blocks_written", 0)
            recs.append(Recommendation(
                category="Performance",
                severity="medium",
                title="High Temp File Usage",
                description=f"Query wrote {temp_blocks:,} blocks to temp files.",
                actions=[
                    "Increase work_mem in postgresql.conf",
                    "Optimize queries to reduce memory requirements",
                    "Consider breaking large operations into batches",
                ],
                confidence=0.88,
            ))
        
        if signal.name == "sequential_scan_heavy":
            percent = signal.data.get("seq_scan_percent", 0)
            tables = signal.data.get("tables", [])
            recs.append(Recommendation(
                category="Performance",
                severity="high",
                title=f"Heavy Sequential Scans: {percent}%",
                description=f"Sequential scans on: {', '.join(tables)}",
                actions=[
                    "Add indexes on columns used in WHERE clauses",
                    "Review and optimize query patterns",
                    "Run ANALYZE to update statistics",
                ],
                confidence=0.88,
            ))
        
        if signal.name == "severe_row_estimation_error":
            error_factor = signal.data.get("error_factor", 0)
            recs.append(Recommendation(
                category="Statistics",
                severity="high",
                title=f"Row Estimation Error: {error_factor}x",
                description="Planner severely misestimated row count.",
                actions=[
                    "Run ANALYZE on affected tables",
                    "Check for skewed data distributions",
                    "Consider creating statistics for specific columns",
                ],
                confidence=0.90,
            ))
        
        return recs
    
    def _table_stats_rules(self, signal: Signal) -> List[Recommendation]:
        """Table statistics rules."""
        recs = []
        
        if signal.name == "high_dead_tuples":
            table_name = signal.data.get("table_name", "unknown")
            dead_tuples = signal.data.get("dead_tuples", 0)
            recs.append(Recommendation(
                category="Maintenance",
                severity="medium",
                title=f"High Dead Tuples: {table_name}",
                description=f"Table has {dead_tuples:,} dead tuples.",
                actions=[
                    f"Run VACUUM (VERBOSE) {table_name}",
                    "Check autovacuum settings for this table",
                    "Consider increasing autovacuum_vacuum_cost_delay",
                ],
                confidence=0.90,
            ))
        
        if signal.name == "stale_table_statistics":
            recs.append(Recommendation(
                category="Statistics",
                severity="medium",
                title="Stale Table Statistics",
                description="Table statistics are outdated.",
                actions=[
                    "Run ANALYZE on affected tables",
                    "Check autovacuum_analyze settings",
                    "Consider manual ANALYZE after bulk operations",
                ],
                confidence=0.85,
            ))
        
        return recs
    
    # ------------------------------------------------------------------
    # DBA-Safe Recommendation Methods
    # ------------------------------------------------------------------
    
    def _action_to_db_action(self, action: Action, priority: str = "immediate") -> DBAction:
        """
        Convert a KB Action to a DBAction with full safety information.
        
        Args:
            action: The KB Action to convert
            priority: Priority label (immediate/long-term)
            
        Returns:
            DBAction with safety information
        """
        # Determine risk level from action
        risk = action.risk.lower() if action.risk else "low"
        if risk not in ["low", "medium", "high"]:
            risk = "medium"  # Default to medium for unknown risks
        
        # Check for approval requirements
        requires_approval = self._check_approval_required(action.sql_example)
        
        # Determine if online operation
        is_online = self._check_online_safe(action.sql_example)
        
        # Get or generate rollback notes
        rollback_notes = action.rollback_notes if action.rollback_notes else \
            self._generate_rollback_notes(action.sql_example)
        
        return DBAction(
            action=action.action,
            sql=action.sql_example,
            risk_level=risk,
            is_online=is_online,
            requires_approval=requires_approval,
            rollback_notes=rollback_notes,
            priority=priority,
        )
    
    def _check_approval_required(self, sql: str) -> bool:
        """
        Check if SQL requires DBA approval.
        
        Rules:
        - DROP operations always require approval
        - TRUNCATE always requires approval
        - VACUUM FULL requires approval (unless justified)
        - pg_terminate_backend requires approval
        - Standard REINDEX requires approval
        - ALTER TABLE REWRITE requires approval
        """
        if not sql:
            return False
        
        sql_upper = sql.upper()
        sql_lower = sql.lower()
        
        # Check for destructive operations
        for op in DESTRUCTIVE_OPERATIONS.keys():
            if op.replace("_", " ") in sql_upper.replace("_", " "):
                # Special handling for VACUUM FULL
                if op == "VACUUM_FULL" and "FULL" in sql_upper:
                    return True
                if op == "REINDEX" and "CONCURRENTLY" not in sql_upper:
                    return True
                if op in ["DROP", "TRUNCATE"]:
                    return True
                return True
        
        # Check for pg_terminate_backend and pg_cancel_backend
        if "pg_terminate_backend" in sql_lower or "pg_cancel_backend" in sql_lower:
            return True
        
        return False
    
    def _check_online_safe(self, sql: str) -> bool:
        """
        Check if SQL can run online (while database is active).
        
        Rules:
        - Most DDL (CREATE INDEX, ANALYZE, etc.) can run online with CONCURRENTLY
        - VACUUM FULL cannot run online
        - ALTER SYSTEM for some parameters requires restart
        """
        if not sql:
            return True
        
        sql_lower = sql.lower()
        
        # VACUUM FULL is not online safe
        if "vacuum full" in sql_lower:
            return False
        
        # ALTER SYSTEM for shared_buffers requires restart
        if "shared_buffers" in sql_lower and "alter system" in sql_lower:
            return False
        
        # Default to online safe
        return True
    
    def _generate_rollback_notes(self, sql: str) -> str:
        """Generate rollback notes for SQL operations."""
        if not sql:
            return "Not applicable - this is a read-only operation"
        
        sql_upper = sql.upper()
        sql_lower = sql.lower()
        
        if "CREATE INDEX" in sql_upper:
            if "CONCURRENTLY" in sql_upper:
                return "DROP INDEX CONCURRENTLY {index_name};"
            return "DROP INDEX {index_name}; (Note: This locks the table)"
        
        if "CREATE INDEX" not in sql_upper and "DROP INDEX" in sql_upper:
            return "Recreate the index using the original definition"
        
        if "ANALYZE" in sql_upper:
            return "Not needed - ANALYZE only collects statistics, no data modification"
        
        if "VACUUM" in sql_upper:
            if "FULL" in sql_upper:
                return "Not needed - VACUUM FULL is a recovery operation, but requires maintenance window"
            return "Not needed - VACUUM only reclaims space and does not modify data"
        
        if "SET" in sql_upper and "work_mem" in sql_lower:
            return "Reset with: RESET work_mem; or restart PostgreSQL to restore default"
        
        if "ALTER SYSTEM" in sql_upper:
            return "To rollback: ALTER SYSTEM SET {parameter} = '{old_value}'; Requires PostgreSQL restart"
        
        if "pg_blocking" in sql_lower or "pg_terminate" in sql_lower:
            return "Let the transaction complete naturally if possible. Terminating should be last resort"
        
        if "EXPLAIN" in sql_upper:
            return "Not applicable - EXPLAIN is a read-only diagnostic operation"
        
        return "Review PostgreSQL documentation for rollback procedures"
    
    def _kb_entry_to_db_recommendation(
        self,
        result: SearchResult,
        priority: str = "immediate"
    ) -> Optional[Recommendation]:
        """Convert KB entry to DBA-safe Recommendation."""
        entry = result.entry
        
        if not entry.get_actionable_recommendations():
            return None
        
        # Convert actions to DBActions
        db_actions = []
        for action in entry.recommendations.immediate_actions:
            db_actions.append(self._action_to_db_action(action, "immediate"))
        
        for action in entry.recommendations.long_term_fixes:
            db_actions.append(self._action_to_db_action(action, "long-term"))
        
        # Determine overall risk level
        risk_levels = [a.risk_level for a in db_actions]
        overall_risk = "low"
        if "high" in risk_levels:
            overall_risk = "high"
        elif "medium" in risk_levels:
            overall_risk = "medium"
        
        # Check if any action requires approval
        requires_approval = any(a.requires_approval for a in db_actions)
        
        return Recommendation(
            category=entry.metadata.category,
            severity=entry.metadata.severity,
            title=f"[KB] {entry.problem_identity.issue_type}",
            description=entry.problem_identity.short_description,
            actions=db_actions,
            confidence=entry.confidence.confidence_score,
            risk_level=overall_risk,
            requires_approval=requires_approval,
            references=entry.get_sql_examples()[:2] + entry.get_config_examples()[:2],
            kb_entry_id=entry.metadata.kb_id,
            match_type=result.match_type,
            evidence=[
                f"Evidence count: {entry.confidence.evidence_count}",
                entry.confidence.confidence_reasoning,
            ],
            root_cause=entry.root_cause_analysis.primary_cause,
        )
    
    def _generate_db_safe_recommendation(
        self,
        signal: Signal,
        root_cause_category: str,
        root_cause_issue: str,
        table_name: str = None,
        index_name: str = None,
        columns: str = None,
        query: str = None,
    ) -> Optional[Recommendation]:
        """
        Generate a DBA-safe recommendation from root cause analysis.
        
        Args:
            signal: The signal that triggered this recommendation
            root_cause_category: Category from RootCauseCategory (e.g., "INDEX_ISSUES")
            root_cause_issue: Specific issue (e.g., "missing_index")
            table_name: Optional table name
            index_name: Optional index name
            columns: Optional column names
            query: Optional query text
            
        Returns:
            DBA-safe Recommendation or None
        """
        category_actions = ROOT_CAUSE_TO_ACTIONS.get(root_cause_category, {})
        action_config = category_actions.get(root_cause_issue)
        
        if not action_config:
            return None
        
        # Fill in template variables
        sql = None
        if action_config.get("sql_template"):
            sql = action_config["sql_template"]
            sql = sql.replace("{table}", table_name or "table_name")
            sql = sql.replace("{columns}", columns or "column_list")
            sql = sql.replace("{index_name}", index_name or "index_name")
            sql = sql.replace("{blocked_pid}", "blocked_pid")
            sql = sql.replace("{query}", query or "your_query")
            sql = sql.replace("{table_name}", table_name or "table_name")
            sql = sql.replace("{value}", "64")  # Default value
            sql = sql.replace("{old_value}", "original_value")
        
        # Create DBAction
        db_action = DBAction(
            action=action_config["action"],
            sql=sql,
            risk_level=action_config["risk"],
            is_online=action_config["is_online"],
            requires_approval=action_config["requires_approval"],
            rollback_notes=action_config["rollback_notes"],
            priority=action_config["priority"],
        )
        
        # Create Recommendation
        severity_map = {
            "low": "low",
            "medium": "medium",
            "high": "high",
        }
        
        return Recommendation(
            category=root_cause_category.lower(),
            severity=severity_map.get(action_config["risk"], "medium"),
            title=f"DBA-Safe: {action_config['action']}",
            description=f"Recommendation for {root_cause_issue.replace('_', ' ')} in {table_name or 'database'}",
            actions=[db_action],
            confidence=signal.confidence if signal.confidence else 0.85,
            risk_level=action_config["risk"],
            requires_approval=action_config["requires_approval"],
            evidence=[f"Signal: {signal.name}", f"Type: {signal.type}"],
            root_cause=f"{root_cause_category}.{root_cause_issue}",
        )
    
    def recommend_with_root_cause(
        self,
        signal_result,
        root_cause_results: Dict,
    ) -> RecommendationReport:
        """
        Generate DBA-safe recommendations based on root cause analysis.
        
        This method combines signal analysis with root cause determination
        to create targeted, DBA-safe recommendations.
        
        Args:
            signal_result: SignalResult from SignalEngine
            root_cause_results: Dict from RootCauseEngine.analyze()
            
        Returns:
            RecommendationReport with DBA-safe recommendations
        """
        recommendations: List[Recommendation] = []
        kb_references: List[Dict[str, Any]] = []
        
        # Process each root cause category
        for category, result in root_cause_results.items():
            if not result.is_likely_cause:
                continue
            
            category_str = category.value
            
            # Map root cause category to action type
            action_type_map = {
                "INDEX_ISSUES": "missing_index",
                "STATISTICS_MAINTENANCE": "stale_statistics",
                "BLOCKING_LOCKING": "blocking_detected",
                "CONFIGURATION": "plan_regression",
                "CAPACITY_HARDWARE": "disk_io_bottleneck",
                "BACKGROUND_JOBS": "autovacuum_interference",
                "APPLICATION_BEHAVIOR": "unbounded_result_set",
                "DEPLOYMENT_SCHEMA": "recent_deployment",
            }
            
            action_type = action_type_map.get(category_str, "missing_index")
            
            # Extract table/index info from signals
            table_name = None
            index_name = None
            columns = None
            
            for signal in signal_result.signals:
                if signal.data:
                    if "table" in signal.data:
                        table_name = signal.data.get("table") or signal.data.get("table_name")
                    if "index" in signal.data:
                        index_name = signal.data.get("index") or signal.data.get("index_name")
                    if "columns" in signal.data:
                        columns = signal.data.get("columns")
            
            # Generate DBA-safe recommendation
            db_rec = self._generate_db_safe_recommendation(
                signal=signal_result.signals[0] if signal_result.signals else None,
                root_cause_category=category_str,
                root_cause_issue=action_type,
                table_name=table_name,
                index_name=index_name,
                columns=columns,
            )
            
            if db_rec:
                recommendations.append(db_rec)
        
        # Also check KB for matching entries
        if self.kb and self.kb.entry_count > 0:
            symptoms = self._extract_symptoms(signal_result)
            
            if self.vector_index and symptoms:
                query_text = " ".join(symptoms[:3])
                semantic_results = self.vector_index.search(query_text, top_k=5)
                
                for result in semantic_results:
                    kb_rec = self._kb_entry_to_db_recommendation(result)
                    if kb_rec:
                        # Avoid duplicates
                        if not any(r.kb_entry_id == kb_rec.kb_entry_id for r in recommendations):
                            recommendations.append(kb_rec)
                            kb_references.append(result.to_dict())
        
        # Calculate overall risk level
        risk_level = self._calculate_risk(signal_result.signals, recommendations)
        
        # Extract actions requiring approval and offline actions
        actions_requiring_approval = []
        offline_actions = []
        
        for rec in recommendations:
            for action in rec.actions:
                if action.requires_approval:
                    actions_requiring_approval.append({
                        "recommendation": rec.title,
                        "action": action.action,
                        "sql": action.sql,
                        "risk": action.risk_level,
                        "rollback": action.rollback_notes,
                    })
                if not action.is_online:
                    offline_actions.append({
                        "recommendation": rec.title,
                        "action": action.action,
                        "sql": action.sql,
                        "requires_restart": not action.is_online,
                        "rollback": action.rollback_notes,
                    })
        
        # Build summary
        summary = self._build_db_safe_summary(
            recommendations,
            risk_level,
            len(kb_references),
            len(actions_requiring_approval),
            len(offline_actions),
        )
        
        return RecommendationReport(
            recommendations=recommendations,
            risk_level=risk_level,
            summary=summary,
            metadata={
                "signal_count": len(signal_result.signals),
                "root_causes_analyzed": len([r for r in root_cause_results.values() if r.is_likely_cause]),
                "kb_entries_used": len(kb_references),
                "dba_safe_mode": True,
            },
            kb_references=kb_references,
            actions_requiring_approval=actions_requiring_approval,
            offline_actions=offline_actions,
        )
    
    def _build_db_safe_summary(
        self,
        recommendations: List[Recommendation],
        risk_level: str,
        kb_count: int,
        approval_count: int,
        offline_count: int,
    ) -> str:
        """Build DBA-safe recommendation summary."""
        if not recommendations:
            return "No issues detected. System appears healthy."
        
        parts = [f"Risk Level: {risk_level}."]
        parts.append(f"{len(recommendations)} recommendation(s) generated.")
        
        if kb_count > 0:
            parts.append(f"({kb_count} KB entries used)")
        
        if approval_count > 0:
            parts.append(f"\n⚠️ {approval_count} action(s) require DBA approval before execution.")
        
        if offline_count > 0:
            parts.append(f"🔧 {offline_count} action(s) require maintenance window (offline operation).")
        
        if risk_level in ["HIGH", "CRITICAL"]:
            parts.append("\nImmediate attention required for high/critical items.")
        
        return " ".join(parts)
    
    def get_action_safety_info(self, action: str) -> Dict[str, Any]:
        """
        Get safety information for a specific action.
        
        Args:
            action: The action description or SQL
            
        Returns:
            Dict with safety information including risk level,
            online/offline status, approval requirements, and rollback notes
        """
        sql_lower = action.lower()
        
        # Initialize safety info
        safety_info = {
            "action": action,
            "risk_level": "low",
            "is_online": True,
            "requires_approval": False,
            "rollback_notes": "",
            "warnings": [],
        }
        
        # Check for destructive operations
        for op, description in DESTRUCTIVE_OPERATIONS.items():
            if op.replace("_", " ").lower() in sql_lower.replace("_", " "):
                safety_info["risk_level"] = "high"
                safety_info["requires_approval"] = True
                safety_info["warnings"].append(f"DESTRUCTIVE: {description}")
        
        # Check for VACUUM FULL
        if "vacuum full" in sql_lower:
            safety_info["is_online"] = False
            safety_info["warnings"].append(
                "VACUUM FULL requires exclusive lock. Use VACUUM (without FULL) for online operation."
            )
            # Check for justification
            justification_found = any(
                just.lower() in sql_lower for just in VACUUM_FULL_JUSTIFICATIONS
            )
            if not justification_found:
                safety_info["warnings"].append(
                    "VACUUM FULL requires explicit justification (severe bloat, corrupted data, etc.)"
                )
        
        # Check for standard REINDEX
        if "reindex" in sql_lower and "concurrently" not in sql_lower:
            safety_info["risk_level"] = "high"
            safety_info["warnings"].append(
                "Standard REINDEX locks writes. Use REINDEX CONCURRENTLY for online operation."
            )
        
        # Check for parameter changes
        if "alter system" in sql_lower or "postgresql.conf" in sql_lower:
            if "shared_buffers" in sql_lower or "work_mem" in sql_lower:
                safety_info["requires_approval"] = True
                safety_info["is_online"] = False
                safety_info["warnings"].append("Parameter change may require PostgreSQL restart")
        
        # Generate rollback notes
        safety_info["rollback_notes"] = self._generate_rollback_notes(action)
        
        return safety_info

