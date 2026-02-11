#!/usr/bin/env python3
"""
Incident Output Schema (v1.0.0)

Production-grade schema for PostgreSQL DBA agent incident output.

Design Principles:
- Deterministic and auditable
- No free-text fields without structure
- Suitable for SREs, DBAs, and management
- Supports confidence scoring
- Integration: Post-processor pattern (IncidentOutputFormatter)

Sections:
1. Incident Summary
2. Root Cause (can be multiple)
3. Evidence (metrics, facts, sources)
4. Impact Analysis
5. Recommended Actions
6. Safety & Approval Notes
7. Confidence Score
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------
# Enums (Deterministic Value Sets)
# ---------------------------------------------------------------------

class IncidentSeverity(Enum):
    """Standard severity levels for incidents."""
    P1_CRITICAL = "P1_CRITICAL"  # Complete outage, data loss
    P2_HIGH = "P2_HIGH"          # Degraded service, workarounds available
    P3_MEDIUM = "P3_MEDIUM"      # Performance issues, minor impact
    P4_LOW = "P4_LOW"            # Minor issues, informational
    P5_INFO = "P5_INFO"          # Informational only


class IncidentStatus(Enum):
    """Incident lifecycle status."""
    DETECTED = "DETECTED"
    INVESTIGATING = "INVESTIGATING"
    IDENTIFIED = "IDENTIFIED"
    MONITORING = "MONITORING"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"


class IncidentCategory(Enum):
    """PostgreSQL incident categories."""
    QUERY_PERFORMANCE = "QUERY_PERFORMANCE"
    LOCKING = "LOCKING"
    INDEX_HEALTH = "INDEX_HEALTH"
    MAINTENANCE = "MAINTENANCE"
    CONFIGURATION = "CONFIGURATION"
    HARDWARE_CAPACITY = "HARDWARE_CAPACITY"
    REPLICATION = "REPLICATION"
    CONNECTION = "CONNECTION"
    STORAGE = "STORAGE"
    SECURITY = "SECURITY"
    APPLICATION_IMPACT = "APPLICATION_IMPACT"


class RootCauseType(Enum):
    """Classification of root cause types."""
    MISSING_INDEX = "MISSING_INDEX"
    STALE_STATISTICS = "STALE_STATISTICS"
    LOCK_CONTENTION = "LOCK_CONTENTION"
    BAD_PLAN = "BAD_PLAN"
    RESOURCE_EXHAUSTION = "RESOURCE_EXHAUSTION"
    CONFIGURATION_MISMATCH = "CONFIGURATION_MISMATCH"
    MAINTENANCE_OVERDUE = "MAINTENANCE_OVERDUE"
    HARDWARE_BOTTLENECK = "HARDWARE_BOTTLENECK"
    QUERY_PATTERN_CHANGE = "QUERY_PATTERN_CHANGE"
    DATA_GROWTH = "DATA_GROWTH"
    EXTERNAL_FACTOR = "EXTERNAL_FACTOR"
    UNKNOWN = "UNKNOWN"


class ActionRisk(Enum):
    """Risk level for recommended actions."""
    LOW = "LOW"         # Safe, reversible, no downtime
    MEDIUM = "MEDIUM"   # Requires planning, minimal downtime
    HIGH = "HIGH"       # Service impact, requires approval
    CRITICAL = "CRITICAL"  # Data risk, full approval required


class EvidenceType(Enum):
    """Types of evidence that can be collected."""
    METRIC = "METRIC"
    FACT = "FACT"
    QUERY = "QUERY"
    LOCK = "LOCK"
    CONFIG = "CONFIG"
    TABLE_STATS = "TABLE_STATS"
    INDEX_STATS = "INDEX_STATS"
    EXECUTION_PLAN = "EXECUTION_PLAN"
    LOG_ENTRY = "LOG_ENTRY"
    SOURCE_REFERENCE = "SOURCE_REFERENCE"


class ImpactScope(Enum):
    """Scope of impact assessment."""
    CRITICAL = "CRITICAL"   # Complete service outage
    HIGH = "HIGH"           # Major functionality affected
    MEDIUM = "MEDIUM"       # Degraded performance
    LOW = "LOW"             # Minor/nuisance impact
    NONE = "NONE"           # No perceptible impact


# ---------------------------------------------------------------------
# Core Data Models
# ---------------------------------------------------------------------

@dataclass
class TimeWindow:
    """Time window for incident analysis."""
    start_time: str  # ISO 8601
    end_time: str    # ISO 8601
    duration_seconds: int
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TimeWindow":
        return cls(**data)


@dataclass
class DatabaseContext:
    """Structured database environment context."""
    database_name: str = ""
    schema_name: str = "public"
    table_name: Optional[str] = None
    index_name: Optional[str] = None
    query_fingerprint: Optional[str] = None
    query_hash: Optional[str] = None
    pg_version: Optional[str] = None
    host: Optional[str] = None
    application_name: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DatabaseContext":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MetricEvidence:
    """A single metric with threshold context."""
    metric_name: str
    value: Union[float, int, str]
    unit: str
    threshold: Optional[Union[float, int]] = None
    operator: str = ">"  # >, <, >=, <=, ==, !=
    is_anomaly: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MetricEvidence":
        return cls(**data)


@dataclass
class Evidence:
    """Single piece of supporting evidence."""
    evidence_id: str = field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:8]}")
    evidence_type: str = EvidenceType.FACT.value
    source_signal: str = ""  # Which signal generated this
    description: str = ""  # Structured description
    metric: Optional[MetricEvidence] = None
    raw_value: Optional[Any] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    provenance: str = ""  # e.g., "pg_stat_statements", "pg_stat_user_tables"
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        if self.metric:
            result["metric"] = self.metric.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Evidence":
        metric_data = data.get("metric")
        metric = MetricEvidence.from_dict(metric_data) if metric_data else None
        return cls(
            evidence_id=data.get("evidence_id", ""),
            evidence_type=data.get("evidence_type", EvidenceType.FACT.value),
            source_signal=data.get("source_signal", ""),
            description=data.get("description", ""),
            metric=metric,
            raw_value=data.get("raw_value"),
            timestamp=data.get("timestamp", ""),
            provenance=data.get("provenance", ""),
        )


@dataclass
class RootCause:
    """A single root cause with classification."""
    cause_id: str = field(default_factory=lambda: f"rc_{uuid.uuid4().hex[:8]}")
    cause_type: str = RootCauseType.UNKNOWN.value
    description: str = ""  # Structured description
    contributing_factors: List[str] = field(default_factory=list)
    causation_chain: List[str] = field(default_factory=list)  # Step-by-step
    evidence_ids: List[str] = field(default_factory=list)  # Supporting evidence
    is_primary: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "RootCause":
        return cls(**data)


@dataclass
class Action:
    """A single recommended action."""
    action_id: str = field(default_factory=lambda: f"act_{uuid.uuid4().hex[:8]}")
    action_type: str = ""  # e.g., "CREATE_INDEX", "VACUUM", "CONFIG_CHANGE"
    description: str = ""
    sql_command: Optional[str] = None
    config_change: Optional[Dict[str, Any]] = None
    risk: str = ActionRisk.LOW.value
    estimated_downtime: Optional[str] = None  # e.g., "0s", "5min", "1hr"
    requiresApproval: bool = False
    approver_role: Optional[str] = None  # e.g., "DBA_LEAD", "SRE_LEAD", "MANAGER"
    priority: str = "high"  # critical, high, medium, low
    validation_command: Optional[str] = None
    rollback_command: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Action":
        return cls(**data)


@dataclass
class ApprovalRequirement:
    """Safety and approval requirements."""
    requires_dba_approval: bool = False
    requires_sre_approval: bool = False
    requires_manager_approval: bool = False
    change_ticket_required: bool = False
    maintenance_window_required: bool = False
    downtime_acceptable: bool = False
    rollback_plan: str = ""
    rollback_command: Optional[str] = None
    risk_summary: str = ""
    caveats: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ApprovalRequirement":
        return cls(**data)


@dataclass
class ImpactMetrics:
    """Quantitative impact metrics."""
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    throughput_drop_percent: Optional[float] = None
    error_rate_percent: Optional[float] = None
    blocked_queries_count: Optional[int] = None
    affected_rows_estimate: Optional[int] = None
    affected_connections_count: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ImpactMetrics":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ImpactAnalysis:
    """Structured impact analysis."""
    blast_radius: str = ImpactScope.MEDIUM.value
    affected_services: List[str] = field(default_factory=list)
    affected_endpoints: List[str] = field(default_factory=list)
    business_impact: str = ""  # Structured, not free-text
    technical_impact: str = ""  # Structured, not free-text
    resource_pressure: List[str] = field(default_factory=list)  # cpu, memory, io, storage
    metrics: ImpactMetrics = field(default_factory=ImpactMetrics)
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result["metrics"] = self.metrics.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ImpactAnalysis":
        metrics_data = data.get("metrics", {})
        return cls(
            blast_radius=data.get("blast_radius", ImpactScope.MEDIUM.value),
            affected_services=data.get("affected_services", []),
            affected_endpoints=data.get("affected_endpoints", []),
            business_impact=data.get("business_impact", ""),
            technical_impact=data.get("technical_impact", ""),
            resource_pressure=data.get("resource_pressure", []),
            metrics=ImpactMetrics.from_dict(metrics_data),
        )


@dataclass
class ConfidenceScore:
    """Confidence scoring with full reasoning."""
    score: float = 0.85  # 0.0 to 1.0
    reasoning: str = ""  # Explain the score
    evidence_count: int = 0
    signal_count: int = 0
    uncertainty_factors: List[str] = field(default_factory=list)
    recommended_verification: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ConfidenceScore":
        return cls(**data)


@dataclass
class IncidentSummary:
    """High-level incident summary (structured, no free-text without schema)."""
    incident_id: str = field(default_factory=lambda: f"INC-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}")
    title: str = ""  # Structured: "{category}: {specific_issue}"
    severity: str = IncidentSeverity.P3_MEDIUM.value
    status: str = IncidentStatus.DETECTED.value
    category: str = IncidentCategory.QUERY_PERFORMANCE.value
    detected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    time_window: Optional[TimeWindow] = None
    db_context: DatabaseContext = field(default_factory=DatabaseContext)
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        if self.time_window:
            result["time_window"] = self.time_window.to_dict()
        result["db_context"] = self.db_context.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "IncidentSummary":
        time_window_data = data.get("time_window")
        time_window = TimeWindow.from_dict(time_window_data) if time_window_data else None
        return cls(
            incident_id=data.get("incident_id", ""),
            title=data.get("title", ""),
            severity=data.get("severity", IncidentSeverity.P3_MEDIUM.value),
            status=data.get("status", IncidentStatus.DETECTED.value),
            category=data.get("category", IncidentCategory.QUERY_PERFORMANCE.value),
            detected_at=data.get("detected_at", ""),
            acknowledged_at=data.get("acknowledged_at"),
            resolved_at=data.get("resolved_at"),
            time_window=time_window,
            db_context=DatabaseContext.from_dict(data.get("db_context", {})),
        )


# ---------------------------------------------------------------------
# Main Incident Output Schema
# ---------------------------------------------------------------------

@dataclass
class IncidentOutput:
    """
    Complete incident output for PostgreSQL DBA agent.
    
    This is the deterministic, auditable output format for incidents.
    All fields are structured - no free-text without schema.
    
    Suitable for:
    - SREs: Clear impact analysis, severity, confidence
    - DBAs: Technical details, evidence, actions
    - Management: Business impact, approval requirements
    """
    
    # Section 1: Incident Summary
    summary: IncidentSummary = field(default_factory=IncidentSummary)
    
    # Section 2: Root Cause Analysis (can be multiple)
    root_causes: List[RootCause] = field(default_factory=list)
    
    # Section 3: Evidence (metrics, facts, sources)
    evidence: List[Evidence] = field(default_factory=list)
    
    # Section 4: Impact Analysis
    impact: ImpactAnalysis = field(default_factory=ImpactAnalysis)
    
    # Section 5: Recommended Actions
    recommended_actions: List[Action] = field(default_factory=list)
    
    # Section 6: Safety & Approval Notes
    safety_approval: ApprovalRequirement = field(default_factory=ApprovalRequirement)
    
    # Section 7: Confidence Score
    confidence: ConfidenceScore = field(default_factory=ConfidenceScore)
    
    # Metadata
    schema_version: str = "1.0.0"
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    generator_version: str = "pg-agent-v1.0.0"
    correlation_id: Optional[str] = None
    
    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "generator_version": self.generator_version,
            "correlation_id": self.correlation_id,
            "summary": self.summary.to_dict(),
            "root_causes": [rc.to_dict() for rc in self.root_causes],
            "evidence": [e.to_dict() for e in self.evidence],
            "impact": self.impact.to_dict(),
            "recommended_actions": [a.to_dict() for a in self.recommended_actions],
            "safety_approval": self.safety_approval.to_dict(),
            "confidence": self.confidence.to_dict(),
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save_to_file(self, filepath: str) -> None:
        """Save incident to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IncidentOutput":
        """Create IncidentOutput from dictionary."""
        summary = IncidentSummary.from_dict(data.get("summary", {}))
        
        root_causes = [
            RootCause.from_dict(rc) 
            for rc in data.get("root_causes", [])
        ]
        
        evidence = [
            Evidence.from_dict(e) 
            for e in data.get("evidence", [])
        ]
        
        impact = ImpactAnalysis.from_dict(data.get("impact", {}))
        
        recommended_actions = [
            Action.from_dict(a) 
            for a in data.get("recommended_actions", [])
        ]
        
        safety_approval = ApprovalRequirement.from_dict(
            data.get("safety_approval", {})
        )
        
        confidence = ConfidenceScore.from_dict(data.get("confidence", {}))
        
        return cls(
            schema_version=data.get("schema_version", "1.0.0"),
            generated_at=data.get("generated_at", ""),
            generator_version=data.get("generator_version", "pg-agent-v1.0.0"),
            correlation_id=data.get("correlation_id"),
            summary=summary,
            root_causes=root_causes,
            evidence=evidence,
            impact=impact,
            recommended_actions=recommended_actions,
            safety_approval=safety_approval,
            confidence=confidence,
        )
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "IncidentOutput":
        """Load incident from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    # -----------------------------------------------------------------
    # Query Methods (for downstream consumers)
    # -----------------------------------------------------------------
    
    def get_primary_root_cause(self) -> Optional[RootCause]:
        """Get the primary (first) root cause."""
        for rc in self.root_causes:
            if rc.is_primary:
                return rc
        return self.root_causes[0] if self.root_causes else None
    
    def get_high_risk_actions(self) -> List[Action]:
        """Get actions that require approval."""
        return [
            a for a in self.recommended_actions
            if a.requiresApproval or a.risk in (ActionRisk.HIGH.value, ActionRisk.CRITICAL.value)
        ]
    
    def get_evidence_by_type(self, evidence_type: str) -> List[Evidence]:
        """Get evidence filtered by type."""
        return [e for e in self.evidence if e.evidence_type == evidence_type]
    
    def get_metric_evidence(self) -> List[Evidence]:
        """Get all metric-based evidence."""
        return self.get_evidence_by_type(EvidenceType.METRIC.value)
    
    def is_confident(self, threshold: float = 0.8) -> bool:
        """Check if confidence meets threshold."""
        return self.confidence.score >= threshold
    
    def get_severity_level(self) -> str:
        """Get severity level for alerting."""
        return self.summary.severity


# ---------------------------------------------------------------------
# Incident Output Formatter (Integration with EvidenceBuilder)
# ---------------------------------------------------------------------

class IncidentOutputFormatter:
    """
    Converts SignalResult + EvidenceCollection into IncidentOutput.
    
    This is the post-processor that integrates with EvidenceBuilder.
    It produces deterministic, auditable incident output.
    """
    
    def __init__(self, generator_version: str = "pg-agent-v1.0.0"):
        self.generator_version = generator_version
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def format(
        self,
        signal_result,
        evidence_collection,
        context: Optional[Dict[str, Any]] = None,
    ) -> IncidentOutput:
        """
        Format SignalResult + EvidenceCollection into IncidentOutput.
        
        Args:
            signal_result: Output from SignalEngine
            evidence_collection: Output from EvidenceBuilder
            context: Optional additional context
            
        Returns:
            IncidentOutput with full structure
        """
        self.logger.info("Formatting incident output")
        
        # Build summary from signals
        summary = self._build_summary(signal_result, evidence_collection, context)
        
        # Extract evidence from collection
        evidence = self._extract_evidence(evidence_collection)
        
        # Build root causes from signal data
        root_causes = self._build_root_causes(signal_result, evidence_collection)
        
        # Build impact analysis
        impact = self._build_impact(signal_result, evidence_collection)
        
        # Build recommended actions from signals
        recommended_actions = self._build_actions(signal_result, evidence_collection)
        
        # Build safety/approval requirements
        safety_approval = self._build_safety_approval(signal_result, recommended_actions)
        
        # Calculate confidence
        confidence = self._calculate_confidence(evidence_collection)
        
        return IncidentOutput(
            summary=summary,
            root_causes=root_causes,
            evidence=evidence,
            impact=impact,
            recommended_actions=recommended_actions,
            safety_approval=safety_approval,
            confidence=confidence,
            generator_version=self.generator_version,
        )
    
    def _build_summary(self, signal_result, evidence_collection, context) -> IncidentSummary:
        """Build incident summary from signals."""
        # Extract key info from signals
        highest_severity = "P3_MEDIUM"
        category = "QUERY_PERFORMANCE"
        title_parts = []
        
        for signal in signal_result.signals:
            if signal.severity in ("critical",):
                highest_severity = "P1_CRITICAL"
            elif signal.severity == "high" and highest_severity != "P1_CRITICAL":
                highest_severity = "P2_HIGH"
            
            # Determine category from signal type
            signal_type_map = {
                "query_metrics": "QUERY_PERFORMANCE",
                "table_stats": "MAINTENANCE",
                "index_health": "INDEX_HEALTH",
                "locking": "LOCKING",
                "incident": "QUERY_PERFORMANCE",
                "pgbench": "QUERY_PERFORMANCE",
            }
            category = signal_type_map.get(signal.type, category)
            
            title_parts.append(signal.name.replace("_", " ").title())
        
        title = f"{category.replace('_', ' ')}: {', '.join(title_parts[:2])}"
        
        return IncidentSummary(
            title=title,
            severity=getattr(IncidentSeverity, highest_severity.replace("-", "_"), IncidentSeverity.P3_MEDIUM).value,
            category=getattr(IncidentCategory, category, IncidentCategory.QUERY_PERFORMANCE).value,
            db_context=DatabaseContext(
                database_name=context.get("database", "unknown") if context else "unknown",
            ),
        )
    
    def _extract_evidence(self, evidence_collection) -> List[Evidence]:
        """Extract structured evidence from EvidenceCollection."""
        evidence_list = []
        
        for ev in evidence_collection.evidence:
            metric_evidence = None
            if ev.metadata.get("metric_value") is not None:
                metric_evidence = MetricEvidence(
                    metric_name=ev.metadata.get("metric_name", "unknown"),
                    value=ev.metadata.get("metric_value"),
                    unit=ev.metadata.get("unit", ""),
                    threshold=ev.metadata.get("threshold"),
                    is_anomaly=ev.metadata.get("is_anomaly", False),
                )
            
            evidence_list.append(Evidence(
                evidence_id=ev.id,
                evidence_type=ev.type.upper(),
                source_signal=ev.source_signal,
                description=ev.content[:200] if ev.content else "",  # Truncate long content
                metric=metric_evidence,
                raw_value=ev.metadata,
                provenance=ev.metadata.get("source", "unknown"),
            ))
        
        return evidence_list
    
    def _build_root_causes(self, signal_result, evidence_collection) -> List[RootCause]:
        """Build root cause analysis from signals."""
        root_causes = []
        
        for i, signal in enumerate(signal_result.signals):
            cause_type = RootCauseType.UNKNOWN.value
            
            # Map signal types to cause types
            if "missing_index" in signal.name.lower() or "sequential" in signal.name.lower():
                cause_type = RootCauseType.MISSING_INDEX.value
            elif "dead_tuples" in signal.name.lower() or "stale" in signal.name.lower():
                cause_type = RootCauseType.STALE_STATISTICS.value
            elif "blocking" in signal.name.lower() or "lock" in signal.name.lower():
                cause_type = RootCauseType.LOCK_CONTENTION.value
            elif "latency" in signal.name.lower() or "deviation" in signal.name.lower():
                cause_type = RootCauseType.BAD_PLAN.value
            elif "memory" in signal.name.lower() or "temp" in signal.name.lower():
                cause_type = RootCauseType.RESOURCE_EXHAUSTION.value
            
            root_causes.append(RootCause(
                cause_type=cause_type,
                description=signal.metadata.get("explain", signal.name),
                evidence_ids=[f"ev_{signal.id}"],
                is_primary=(i == 0),
            ))
        
        return root_causes
    
    def _build_impact(self, signal_result, evidence_collection) -> ImpactAnalysis:
        """Build impact analysis from signals."""
        # Determine blast radius from severity
        severity = "medium"
        for signal in signal_result.signals:
            if signal.severity == "critical":
                severity = "critical"
                break
            elif signal.severity == "high":
                severity = "high"
        
        scope_map = {
            "critical": ImpactScope.CRITICAL,
            "high": ImpactScope.HIGH,
            "medium": ImpactScope.MEDIUM,
            "low": ImpactScope.LOW,
        }
        
        return ImpactAnalysis(
            blast_radius=scope_map.get(severity, ImpactScope.MEDIUM).value,
            technical_impact=f"PostgreSQL {severity} severity incident detected via {len(signal_result.signals)} signals",
            metrics=ImpactMetrics(),
        )
    
    def _build_actions(self, signal_result, evidence_collection) -> List[Action]:
        """Build recommended actions from signals."""
        actions = []
        
        for signal in signal_result.signals:
            # Generate action based on signal type
            if "missing_index" in signal.name.lower():
                actions.append(Action(
                    action_type="CREATE_INDEX",
                    description=f"Create index on table referenced in signal '{signal.name}'",
                    sql_command="-- Review query pattern and create appropriate index\nCREATE INDEX CONCURRENTLY ...",
                    risk=ActionRisk.MEDIUM.value,
                    estimated_downtime="0s",
                    priority="high",
                ))
            elif "sequential" in signal.name.lower():
                actions.append(Action(
                    action_type="CREATE_INDEX",
                    description="Create index to eliminate sequential scan",
                    risk=ActionRisk.MEDIUM.value,
                    priority="high",
                ))
            elif "dead_tuples" in signal.name.lower():
                actions.append(Action(
                    action_type="VACUUM",
                    description="Run VACUUM to reclaim space and update statistics",
                    sql_command="VACUUM ANALYZE table_name;",
                    risk=ActionRisk.LOW.value,
                    estimated_downtime="0s",
                    priority="medium",
                ))
            elif "blocking" in signal.name.lower():
                actions.append(Action(
                    action_type="TERMINATE_BLOCKING",
                    description="Identify and terminate blocking session",
                    sql_command="SELECT pg_terminate_backend(blocking_pid);",
                    risk=ActionRisk.HIGH.value,
                    requiresApproval=True,
                    approver_role="DBA_LEAD",
                    priority="critical",
                ))
        
        return actions
    
    def _build_safety_approval(self, signal_result, actions: List[Action]) -> ApprovalRequirement:
        """Build safety and approval requirements."""
        requires_approval = any(
            a.risk in (ActionRisk.HIGH.value, ActionRisk.CRITICAL.value) 
            for a in actions
        )
        
        high_risk_actions = [a for a in actions if a.risk == ActionRisk.CRITICAL.value]
        
        risk_summary = "Low risk changes only" if not high_risk_actions else \
            f"Contains {len(high_risk_actions)} high-risk action(s) requiring approval"
        
        return ApprovalRequirement(
            requires_dba_approval=requires_approval,
            requires_sre_approval=any(a.risk == ActionRisk.HIGH.value for a in actions),
            change_ticket_required=requires_approval,
            downtime_acceptable=any(a.risk == ActionRisk.LOW.value for a in actions),
            risk_summary=risk_summary,
            caveats=["Always test in non-production first", "Have rollback plan ready"],
        )
    
    def _calculate_confidence(self, evidence_collection) -> ConfidenceScore:
        """Calculate confidence score from evidence."""
        score = evidence_collection.overall_confidence
        evidence_count = len(evidence_collection.evidence)
        signal_count = evidence_collection.signal_count
        
        # Adjust for evidence quality
        if evidence_count < 3:
            score = max(0.5, score - 0.1)
        
        uncertainty_factors = []
        if evidence_count < 3:
            uncertainty_factors.append("Limited evidence count")
        if score < 0.8:
            uncertainty_factors.append("Below ideal confidence threshold")
        
        reasoning = f"Based on {evidence_count} evidence items from {signal_count} signals"
        
        return ConfidenceScore(
            score=round(score, 2),
            reasoning=reasoning,
            evidence_count=evidence_count,
            signal_count=signal_count,
            uncertainty_factors=uncertainty_factors,
            recommended_verification=["Review evidence in context", "Cross-reference with monitoring"] if score < 0.9 else [],
        )

