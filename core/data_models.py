"""
Core Data Models - Single source of truth for all data structures.

All classes defined here are imported by other modules to eliminate
duplication and ensure consistency. This is the central contract
that all modules adhere to.

Principles:
  1. One definition per concept (no duplicates)
  2. Immutable where possible (frozen dataclasses)
  3. Comprehensive type hints
  4. Explicit None vs missing distinction
  5. Built-in serialization support
"""

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from core.enums import (
    Severity,
    Category,
    Component,
    RootCauseCategory,
    EvidenceType,
    QuestionCategory,
    QuestionPriority,
    ActionRisk,
    ActionApprovalStatus,
)


# =====================================================================
# PRIMITIVE & CONTAINER TYPES
# =====================================================================

@dataclass
class MetricValue:
    """A single metric measurement with metadata."""
    name: str                                    # e.g., "query_latency_ms"
    value: Union[int, float, str]               # The actual value
    unit: str = ""                              # e.g., "ms", "MB", "%"
    threshold: Optional[Union[int, float]] = None  # Anomaly threshold
    is_anomaly: bool = False                    # True if exceeds threshold
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricValue":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =====================================================================
# DATABASE CONTEXT
# =====================================================================

@dataclass
class DatabaseEnvironment:
    """Database and host information."""
    database_name: str = ""
    schema_name: str = "public"
    host: str = ""
    pg_version: str = ""
    application_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatabaseEnvironment":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class QueryFingerprint:
    """Normalized query information."""
    query_hash: str = ""
    normalized_query: str = ""
    query_type: str = ""              # SELECT, INSERT, UPDATE, DELETE, etc.
    tables: List[str] = field(default_factory=list)
    indexes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryFingerprint":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Context:
    """Full context where an incident occurred."""
    environment: DatabaseEnvironment = field(default_factory=DatabaseEnvironment)
    query: QueryFingerprint = field(default_factory=QueryFingerprint)
    tables_involved: List[str] = field(default_factory=list)
    indexes_involved: List[str] = field(default_factory=list)
    affected_rows_estimate: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "environment": self.environment.to_dict(),
            "query": self.query.to_dict(),
            "tables_involved": self.tables_involved,
            "indexes_involved": self.indexes_involved,
            "affected_rows_estimate": self.affected_rows_estimate,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Context":
        return cls(
            environment=DatabaseEnvironment.from_dict(data.get("environment", {})),
            query=QueryFingerprint.from_dict(data.get("query", {})),
            tables_involved=data.get("tables_involved", []),
            indexes_involved=data.get("indexes_involved", []),
            affected_rows_estimate=data.get("affected_rows_estimate"),
            metadata=data.get("metadata", {}),
        )


# =====================================================================
# IMPACT METRICS
# =====================================================================

@dataclass
class ImpactMetrics:
    """Quantitative impact of the incident."""
    # Latency metrics
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    latency_increase_percent: Optional[float] = None

    # Throughput metrics
    throughput_drop_percent: Optional[float] = None
    error_rate_percent: Optional[float] = None
    requests_failed_count: Optional[int] = None

    # Resource impact
    blocked_queries_count: Optional[int] = None
    affected_connections_count: Optional[int] = None
    affected_rows_estimate: Optional[int] = None
    duration_seconds: Optional[float] = None

    # Qualitative impact
    impact_summary: str = ""
    blast_radius: str = ""  # "single_query" | "table" | "database" | "cluster"
    business_impact: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, omitting None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImpactMetrics":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =====================================================================
# EVIDENCE
# =====================================================================

@dataclass
class Evidence:
    """A single piece of supporting evidence.
    
    Unified evidence model used across all modules:
    - signal_engine → evidence_builder creates Evidence objects
    - root_cause_engine uses Evidence for rule matching
    - clarification_engine identifies missing Evidence types
    """
    evidence_id: str = field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:8]}")
    evidence_type: EvidenceType = EvidenceType.FACT
    description: str = ""              # Human-readable explanation
    metric: Optional[MetricValue] = None  # If evidence is metric-based
    confidence: float = 0.5            # 0.0-1.0
    source_signal: str = ""            # Which signal produced this
    raw_value: Optional[Any] = None    # Original unprocessed value
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    provenance: str = ""               # e.g., "pg_stat_statements", "pg_stat_user_tables"
    related_evidence_ids: List[str] = field(default_factory=list)  # IDs of related evidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, handling nested objects."""
        result = asdict(self)
        if self.evidence_type:
            result["evidence_type"] = self.evidence_type.value
        if self.metric:
            result["metric"] = self.metric.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evidence":
        """Create from dict, reconstructing nested objects."""
        metric_data = data.get("metric")
        metric = MetricValue.from_dict(metric_data) if metric_data else None
        
        evidence_type_str = data.get("evidence_type", "fact")
        if isinstance(evidence_type_str, str):
            try:
                evidence_type = EvidenceType(evidence_type_str)
            except (ValueError, KeyError):
                evidence_type = EvidenceType.FACT
        else:
            evidence_type = evidence_type_str
        
        return cls(
            evidence_id=data.get("evidence_id", f"ev_{uuid.uuid4().hex[:8]}"),
            evidence_type=evidence_type,
            description=data.get("description", ""),
            metric=metric,
            confidence=data.get("confidence", 0.5),
            source_signal=data.get("source_signal", ""),
            raw_value=data.get("raw_value"),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            provenance=data.get("provenance", ""),
            related_evidence_ids=data.get("related_evidence_ids", []),
        )


# =====================================================================
# CONFIDENCE
# =====================================================================

@dataclass
class Confidence:
    """Confidence score with detailed breakdown.
    
    Provides both overall score and component breakdown for explainability.
    """
    overall_score: float = 0.5         # 0.0-1.0 final confidence
    base_confidence: float = 0.5        # Base from direct evidence
    evidence_completeness: float = 1.0  # Penalty for missing evidence
    signal_agreement: float = 1.0       # Boost from multiple signals agreeing
    data_freshness: float = 1.0        # Penalty for stale data
    conflict_penalty: float = 1.0      # Penalty for conflicting signals
    
    reasoning: str = ""                # Why this confidence score
    evidence_count: int = 0            # Number of evidence items
    signal_count: int = 0              # Number of signals
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Confidence":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def is_high_confidence(self, threshold: float = 0.75) -> bool:
        """Check if confidence exceeds threshold."""
        return self.overall_score >= threshold

    def is_low_confidence(self, threshold: float = 0.50) -> bool:
        """Check if confidence is below threshold."""
        return self.overall_score < threshold


# =====================================================================
# ACTIONS & RECOMMENDATIONS
# =====================================================================

@dataclass
class Action:
    """A recommended action to resolve or mitigate the issue.
    
    Unified Action class used in both KB and incident responses.
    Includes DBA-safety information.
    """
    action_id: str = field(default_factory=lambda: f"act_{uuid.uuid4().hex[:8]}")
    action_type: str = ""              # e.g., "CREATE_INDEX", "VACUUM", "ANALYZE", "CONFIG_CHANGE"
    description: str = ""
    
    # Action details
    title: str = ""
    sql_command: Optional[str] = None
    config_change: Optional[Dict[str, Any]] = None
    priority: str = "high"             # critical, high, medium, low
    
    # DBA Safety
    risk: ActionRisk = ActionRisk.MEDIUM
    requires_approval: bool = False
    approver_role: Optional[str] = None  # e.g., "DBA_LEAD", "SRE_LEAD"
    
    # Operational
    estimated_downtime: Optional[str] = None  # e.g., "0s", "5min", "1hr"
    is_online: bool = True             # Can run without maintenance window
    
    # Validation & Rollback
    validation_command: Optional[str] = None
    rollback_command: Optional[str] = None
    
    # References
    kb_entry_id: Optional[str] = None
    references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, handling enums."""
        result = asdict(self)
        if isinstance(self.risk, ActionRisk):
            result["risk"] = self.risk.value
        return {k: v for k, v in result.items() if v is not None or k in ["risk", "is_online", "requires_approval"]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Create from dict."""
        risk_str = data.get("risk", "medium")
        if isinstance(risk_str, str):
            try:
                risk = ActionRisk(risk_str)
            except (ValueError, KeyError):
                risk = ActionRisk.MEDIUM
        else:
            risk = risk_str

        return cls(
            action_id=data.get("action_id", f"act_{uuid.uuid4().hex[:8]}"),
            action_type=data.get("action_type", ""),
            description=data.get("description", ""),
            title=data.get("title", ""),
            sql_command=data.get("sql_command"),
            config_change=data.get("config_change"),
            priority=data.get("priority", "high"),
            risk=risk,
            requires_approval=data.get("requires_approval", False),
            approver_role=data.get("approver_role"),
            estimated_downtime=data.get("estimated_downtime"),
            is_online=data.get("is_online", True),
            validation_command=data.get("validation_command"),
            rollback_command=data.get("rollback_command"),
            kb_entry_id=data.get("kb_entry_id"),
            references=data.get("references", []),
        )


# =====================================================================
# ROOT CAUSE
# =====================================================================

@dataclass
class RootCauseDetail:
    """Details of a detected root cause."""
    category: RootCauseCategory
    title: str = ""
    description: str = ""
    confidence: float = 0.5
    contributing_factors: List[str] = field(default_factory=list)
    causation_steps: List[str] = field(default_factory=list)
    false_positive_warnings: List[str] = field(default_factory=list)
    supporting_evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["category"] = self.category.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RootCauseDetail":
        category_str = data.get("category", "index_issues")
        if isinstance(category_str, str):
            try:
                category = RootCauseCategory(category_str)
            except (ValueError, KeyError):
                category = RootCauseCategory.INDEX_ISSUES
        else:
            category = category_str

        return cls(
            category=category,
            title=data.get("title", ""),
            description=data.get("description", ""),
            confidence=data.get("confidence", 0.5),
            contributing_factors=data.get("contributing_factors", []),
            causation_steps=data.get("causation_steps", []),
            false_positive_warnings=data.get("false_positive_warnings", []),
            supporting_evidence_ids=data.get("supporting_evidence_ids", []),
        )


# =====================================================================
# CLARIFICATION QUESTIONS
# =====================================================================

@dataclass
class ClarificationQuestion:
    """A clarification question to improve confidence."""
    question_id: str = field(default_factory=lambda: f"q_{uuid.uuid4().hex[:8]}")
    template_id: str = ""              # e.g., "DEP_001", "PERF_002"
    category: QuestionCategory = QuestionCategory.DEPLOYMENT
    priority: QuestionPriority = QuestionPriority.HIGH
    
    question_text: str = ""
    expected_evidence_type: EvidenceType = EvidenceType.FACT
    answer_options: List[str] = field(default_factory=list)
    confidence_impact_percent: float = 10.0  # Expected confidence improvement
    
    answer: Optional[str] = None
    answered_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["category"] = self.category.value
        result["priority"] = self.priority.value
        result["expected_evidence_type"] = self.expected_evidence_type.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClarificationQuestion":
        return cls(
            question_id=data.get("question_id", f"q_{uuid.uuid4().hex[:8]}"),
            template_id=data.get("template_id", ""),
            category=QuestionCategory(data.get("category", "deployment")),
            priority=QuestionPriority(data.get("priority", "high")),
            question_text=data.get("question_text", ""),
            expected_evidence_type=EvidenceType(data.get("expected_evidence_type", "fact")),
            answer_options=data.get("answer_options", []),
            confidence_impact_percent=data.get("confidence_impact_percent", 10.0),
            answer=data.get("answer"),
            answered_at=data.get("answered_at"),
        )

    def is_answered(self) -> bool:
        """Check if question has been answered."""
        return self.answer is not None and len(self.answer) > 0
