#!/usr/bin/env python3
"""
kb_schema.py

PostgreSQL Knowledge Base Schema Definition

Formal schema for KB entries used in the recommendation engine.
This provides structure for:
- Problem identification
- Root cause analysis
- Impact assessment
- Actionable recommendations
- Evidence and confidence scoring

Designed for:
- Rule-based retrieval (deterministic matching)
- Similarity-based retrieval (vector embeddings)
- Hybrid retrieval (combine both approaches)
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

# Import enums from core/enums for consistency across the codebase
from core.enums import Severity, Category, Component


@dataclass
class Metadata:
    """KB entry metadata."""
    kb_id: str
    category: str
    severity: str
    source: str
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Metadata":
        return cls(**data)


@dataclass
class QueryFingerprint:
    """Normalized query information."""
    query_hash: str = ""
    normalized_query: str = ""
    query_type: str = ""  # SELECT, INSERT, UPDATE, DELETE
    tables: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "QueryFingerprint":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Environment:
    """Environment information."""
    database: str = ""
    schema: str = "public"
    application: str = ""
    host: str = ""
    pg_version: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Environment":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Context:
    """Context information where the issue occurred."""
    query_fingerprint: QueryFingerprint = field(default_factory=QueryFingerprint)
    environment: Environment = field(default_factory=Environment)
    tables_involved: List[str] = field(default_factory=list)
    indexes_involved: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "query_fingerprint": self.query_fingerprint.to_dict(),
            "environment": self.environment.to_dict(),
            "tables_involved": self.tables_involved,
            "indexes_involved": self.indexes_involved,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Context":
        return cls(
            query_fingerprint=QueryFingerprint.from_dict(data.get("query_fingerprint", {})),
            environment=Environment.from_dict(data.get("environment", {})),
            tables_involved=data.get("tables_involved", []),
            indexes_involved=data.get("indexes_involved", []),
        )


@dataclass
class DetectionMetrics:
    """Key performance metrics."""
    # Query metrics
    latency_deviation_factor: Optional[float] = None
    mean_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None
    calls: Optional[int] = None
    temp_files_created: Optional[int] = None
    cache_hit_ratio: Optional[float] = None
    
    # Table metrics
    dead_tuple_ratio: Optional[float] = None
    days_since_analyze: Optional[int] = None
    days_since_vacuum: Optional[int] = None
    
    # Index metrics
    idx_scan: Optional[int] = None
    seq_scan: Optional[int] = None
    bloat_ratio: Optional[float] = None
    
    # Lock metrics
    wait_duration_seconds: Optional[int] = None
    blocked_pid: Optional[int] = None
    blocking_pid: Optional[int] = None
    
    # Configuration metrics
    work_mem_mb: Optional[int] = None
    shared_buffers_mb: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DetectionMetrics":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DetectionSignals:
    """How the issue was detected."""
    metrics: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, str] = field(default_factory=dict)
    anomaly_flags: Dict[str, bool] = field(default_factory=dict)
    source_queries: List[str] = field(default_factory=list)
    detection_method: str = "threshold_alert"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DetectionSignals":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RootCauseAnalysis:
    """Why did the issue happen."""
    primary_cause: str = ""
    contributing_factors: List[str] = field(default_factory=list)
    causation_chain: List[str] = field(default_factory=list)
    planner_misbehavior: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "RootCauseAnalysis":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ImpactAnalysis:
    """What was affected."""
    latency_impact: str = ""
    throughput_impact: str = ""
    resource_pressure: List[str] = field(default_factory=list)
    blast_radius: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ImpactAnalysis":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Action:
    """
    A single action to take for DBA-safe recommendations.
    
    Extended with DBA-safety fields:
    - is_online: Whether action can run while database is active
    - requires_approval: Whether DBA approval is required
    - rollback_notes: Instructions for rollback if issues occur
    """
    action: str = ""
    sql_example: str = ""
    config_example: str = ""
    risk: str = "low"  # low/medium/high
    estimated_downtime: str = ""
    priority: str = ""
    # DBA-safe fields
    is_online: bool = True  # Can run while database is active
    requires_approval: bool = False  # Requires DBA approval
    rollback_notes: str = ""  # How to rollback this action
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v or k in ['risk', 'is_online', 'requires_approval']}
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Action":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Recommendations:
    """
    How to fix the issue.
    
    Each action now includes DBA-safety information:
    - Risk level (low/medium/high)
    - Online vs Offline operation
    - Approval requirements
    - Rollback instructions
    """
    immediate_actions: List[Action] = field(default_factory=list)
    long_term_fixes: List[Action] = field(default_factory=list)
    validation_steps: List[str] = field(default_factory=list)
    preventive_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "immediate_actions": [a.to_dict() for a in self.immediate_actions],
            "long_term_fixes": [a.to_dict() for a in self.long_term_fixes],
            "validation_steps": self.validation_steps,
            "preventive_actions": self.preventive_actions,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Recommendations":
        return cls(
            immediate_actions=[Action.from_dict(a) for a in data.get("immediate_actions", [])],
            long_term_fixes=[Action.from_dict(a) for a in data.get("long_term_fixes", [])],
            validation_steps=data.get("validation_steps", []),
            preventive_actions=data.get("preventive_actions", []),
        )
    
    def get_all_actions(self) -> List[Action]:
        """Get all actions (immediate + long-term)."""
        return self.immediate_actions + self.long_term_fixes
    
    def get_actions_requiring_approval(self) -> List[Action]:
        """Get actions that require DBA approval."""
        return [a for a in self.get_all_actions() if a.requires_approval]
    
    def get_offline_actions(self) -> List[Action]:
        """Get actions that require offline/maintenance window."""
        return [a for a in self.get_all_actions() if not a.is_online]
    
    def get_high_risk_actions(self) -> List[Action]:
        """Get high-risk actions."""
        return [a for a in self.get_all_actions() if a.risk == "high"]


@dataclass
class Evidence:
    """Supporting data and metrics."""
    query_metrics: Dict[str, Any] = field(default_factory=dict)
    table_statistics: List[Dict[str, Any]] = field(default_factory=list)
    index_statistics: List[Dict[str, Any]] = field(default_factory=list)
    locking_details: Dict[str, Any] = field(default_factory=dict)
    configuration_values: Dict[str, Any] = field(default_factory=dict)
    hardware_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Evidence":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Confidence:
    """Reliability metrics for this diagnosis."""
    confidence_score: float = 0.85
    confidence_reasoning: str = ""
    evidence_count: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Confidence":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ProblemIdentity:
    """What is the issue."""
    issue_type: str = ""
    short_description: str = ""
    long_description: str = ""
    symptoms: List[str] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ProblemIdentity":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class KBEntry:
    """
    A single knowledge base entry for PostgreSQL troubleshooting.
    
    This is the core data structure used by the recommender engine.
    It can be used for:
    - Rule-based matching (exact/partial attribute matching)
    - Similarity-based matching (vector embeddings)
    - Hybrid retrieval (combine both approaches)
    """
    metadata: Metadata
    problem_identity: ProblemIdentity
    detection_signals: DetectionSignals
    context: Context
    root_cause_analysis: RootCauseAnalysis
    recommendations: Recommendations
    evidence: Evidence
    confidence: Confidence
    impact_analysis: ImpactAnalysis = field(default_factory=ImpactAnalysis)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "problem_identity": self.problem_identity.to_dict(),
            "detection_signals": self.detection_signals.to_dict(),
            "context": self.context.to_dict(),
            "root_cause_analysis": self.root_cause_analysis.to_dict(),
            "recommendations": self.recommendations.to_dict(),
            "evidence": self.evidence.to_dict(),
            "confidence": self.confidence.to_dict(),
            "impact_analysis": self.impact_analysis.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "KBEntry":
        """Create KBEntry from dictionary."""
        return cls(
            metadata=Metadata.from_dict(data.get("metadata", {})),
            problem_identity=ProblemIdentity.from_dict(data.get("problem_identity", {})),
            detection_signals=DetectionSignals.from_dict(data.get("detection_signals", {})),
            context=Context.from_dict(data.get("context", {})),
            root_cause_analysis=RootCauseAnalysis.from_dict(data.get("root_cause_analysis", {})),
            recommendations=Recommendations.from_dict(data.get("recommendations", {})),
            evidence=Evidence.from_dict(data.get("evidence", {})),
            confidence=Confidence.from_dict(data.get("confidence", {})),
            impact_analysis=ImpactAnalysis.from_dict(data.get("impact_analysis", {})),
        )
    
    def get_text_content(self) -> str:
        """Get all text content for embedding generation."""
        parts = [
            self.problem_identity.short_description,
            self.problem_identity.long_description,
            " ".join(self.problem_identity.symptoms),
            " ".join(self.problem_identity.affected_components),
            self.root_cause_analysis.primary_cause,
            " ".join(self.root_cause_analysis.contributing_factors),
            " ".join(self.root_cause_analysis.causation_chain),
            self.impact_analysis.latency_impact,
            " ".join(self.impact_analysis.resource_pressure),
        ]
        
        # Add recommendation text
        for action in self.recommendations.immediate_actions:
            parts.append(action.action)
        for action in self.recommendations.long_term_fixes:
            parts.append(action.action)
        parts.extend(self.recommendations.validation_steps)
        parts.extend(self.recommendations.preventive_actions)
        
        return " ".join(p for p in parts if p)
    
    def get_searchable_text(self) -> str:
        """Get searchable text for keyword matching."""
        parts = [
            self.metadata.kb_id,
            self.metadata.category,
            self.metadata.severity,
            " ".join(self.metadata.tags),
            self.problem_identity.issue_type,
            self.problem_identity.short_description,
            self.root_cause_analysis.primary_cause,
        ]
        return " ".join(p for p in parts if p)
    
    def matches_category(self, category: str) -> bool:
        """Check if entry matches a category."""
        return self.metadata.category.lower() == category.lower()
    
    def matches_severity(self, severity: str) -> bool:
        """Check if entry matches a severity level."""
        return self.metadata.severity.lower() == severity.lower()
    
    def matches_symptom(self, symptom: str) -> bool:
        """Check if entry mentions a symptom."""
        symptom_lower = symptom.lower()
        for s in self.problem_identity.symptoms:
            if symptom_lower in s.lower():
                return True
        return False
    
    def matches_cause(self, cause: str) -> bool:
        """Check if entry mentions a cause."""
        cause_lower = cause.lower()
        text = self.root_cause_analysis.primary_cause
        for factor in self.root_cause_analysis.contributing_factors:
            text += " " + factor
        return cause_lower in text.lower()
    
    def matches_table(self, table: str) -> bool:
        """Check if entry involves a specific table."""
        return table in self.context.tables_involved
    
    def get_actionable_recommendations(self) -> List[str]:
        """Get all actionable recommendations."""
        recs = []
        for action in self.recommendations.immediate_actions:
            if action.action:
                recs.append(action.action)
        for action in self.recommendations.long_term_fixes:
            if action.action:
                recs.append(action.action)
        return recs
    
    def get_sql_examples(self) -> List[str]:
        """Get all SQL examples."""
        sqls = []
        for action in self.recommendations.immediate_actions:
            if action.sql_example:
                sqls.append(action.sql_example)
        for action in self.recommendations.long_term_fixes:
            if action.sql_example:
                sqls.append(action.sql_example)
        return sqls
    
    def get_config_examples(self) -> List[str]:
        """Get all configuration examples."""
        configs = []
        for action in self.recommendations.immediate_actions:
            if action.config_example:
                configs.append(action.config_example)
        for action in self.recommendations.long_term_fixes:
            if action.config_example:
                configs.append(action.config_example)
        return configs


@dataclass
class KBVersion:
    """Knowledge base version information."""
    kb_version: str = "1.0.0"
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    entry_count: int = 0
    entries: List[KBEntry] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "kb_version": self.kb_version,
            "generated_at": self.generated_at,
            "entry_count": self.entry_count,
            "entries": [e.to_dict() for e in self.entries],
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "KBVersion":
        return cls(
            kb_version=data.get("kb_version", "1.0.0"),
            generated_at=data.get("generated_at", ""),
            entry_count=data.get("entry_count", 0),
            entries=[KBEntry.from_dict(e) for e in data.get("entries", [])],
        )
    
    def save_to_file(self, filepath: str) -> None:
        """Save KB to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "KBVersion":
        """Load KB from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def create_kb_id(source_name: str) -> str:
    """Create a unique KB ID."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"kb_{source_name}_{timestamp}"


def hash_text(text: str) -> str:
    """Create a hash for text."""
    return hashlib.md5(text.encode()).hexdigest()

