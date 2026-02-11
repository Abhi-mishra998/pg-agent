#!/usr/bin/env python3
"""
Evidence Card - Data Models for Evidence Display

Provides structured models for rendering evidence cards with
metric details, baselines, deviations, and provenance.
"""

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from signals.evidence_types import (
    EvidenceSource,
    EvidenceType,
    DeviationSeverity,
    EvidenceStatus,
    format_deviation,
    calculate_deviation_severity,
)


# =====================================================================
# EVIDENCE CARD DATA MODELS
# =====================================================================

@dataclass
class MetricValue:
    """A metric measurement with unit and threshold context."""
    name: str                              # e.g., "seq_scan_count"
    value: Union[int, float, str]          # The actual value
    unit: str = ""                         # e.g., "scans", "ms", "bytes"
    threshold: Optional[Union[int, float]] = None  # Anomaly threshold
    operator: str = ">"                    # >, <, >=, <=, ==, !=
    is_anomaly: bool = False               # True if exceeds threshold
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricValue":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Baseline:
    """Expected/baseline value for comparison."""
    value: Union[int, float, str]          # Baseline value
    source: str = "historical"              # "historical", "recommended", "config"
    time_range: Optional[str] = None        # e.g., "last 7 days"
    description: Optional[str] = None       # e.g., "95th percentile"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Baseline":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Deviation:
    """Deviation from baseline with severity and formatting."""
    current_value: Union[int, float, str]   # Actual measured value
    baseline_value: Union[int, float, str]  # Expected/normal value
    absolute: float = 0.0                   # Absolute difference
    percent: float = 0.0                    # Percentage change
    severity: DeviationSeverity = DeviationSeverity.NORMAL
    formatted_percent: str = "0%"           # Human-readable percentage
    is_increase_bad: bool = True            # Whether increase indicates problem
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["severity"] = self.severity.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Deviation":
        severity = data.get("severity", "normal")
        if isinstance(severity, str):
            severity = DeviationSeverity(severity)
        return cls(
            current_value=data.get("current_value", 0),
            baseline_value=data.get("baseline_value", 0),
            absolute=data.get("absolute", 0),
            percent=data.get("percent", 0),
            severity=severity,
            formatted_percent=data.get("formatted_percent", "0%"),
            is_increase_bad=data.get("is_increase_bad", True),
        )
    
    @classmethod
    def calculate(
        cls,
        current_value: Union[int, float],
        baseline_value: Union[int, float],
        is_increase_bad: bool = True
    ) -> "Deviation":
        """Create a Deviation from current and baseline values."""
        if baseline_value == 0:
            absolute = float(current_value)
            if current_value == 0:
                percent = 0.0
                severity = DeviationSeverity.NORMAL
            else:
                percent = float('inf') if current_value > 0 else float('-inf')
                severity = DeviationSeverity.CRITICAL
        else:
            absolute = float(current_value) - float(baseline_value)
            percent = (absolute / float(baseline_value)) * 100
            severity = calculate_deviation_severity(
                float(current_value),
                float(baseline_value),
                is_increase_bad
            )
        
        if percent > 0:
            sign = "+"
        else:
            sign = ""
        
        return cls(
            current_value=current_value,
            baseline_value=baseline_value,
            absolute=absolute,
            percent=percent,
            severity=severity,
            formatted_percent=f"{sign}{percent:,.0f}%",
            is_increase_bad=is_increase_bad,
        )


@dataclass
class EvidenceProvenance:
    """Source information for an evidence item."""
    source: EvidenceSource                  # Source enum
    source_name: str = ""                   # Display name (derived from enum)
    icon: str = "❓"                         # Icon (derived from enum)
    color: str = "gray"                     # Color (derived from enum)
    query: Optional[str] = None              # Query that produced this
    timestamp: str = ""                     # When data was collected
    data_age_hours: Optional[float] = None  # Age of data in hours
    freshness_status: str = "fresh"         # "fresh", "stale", "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["source"] = self.source.value if isinstance(self.source, EvidenceSource) else self.source
        return result
    
    @classmethod
    def from_source(cls, source: EvidenceSource, timestamp: str = "") -> "EvidenceProvenance":
        """Create from EvidenceSource enum."""
        return cls(
            source=source,
            source_name=source.display_name,
            icon=source.icon,
            color=source.color,
            timestamp=timestamp,
        )


@dataclass
class EvidenceCard:
    """
    Complete evidence card for display.
    
    This is the main data model for evidence cards in the Evidence Explorer.
    Every piece of supporting evidence is represented as an EvidenceCard.
    """
    # Core identification
    evidence_id: str = field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:8]}")
    evidence_type: EvidenceType = EvidenceType.METRIC
    
    # Content
    title: str = ""                         # Short title (metric name)
    description: str = ""                   # Human-readable description
    
    # Metric data
    metric: Optional[MetricValue] = None
    baseline: Optional[Baseline] = None
    deviation: Optional[Deviation] = None
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context
    related_evidence_ids: List[str] = field(default_factory=list)  # Cross-references
    
    # Confidence and status
    confidence: float = 0.5                 # 0.0-1.0
    status: EvidenceStatus = EvidenceStatus.CONFIRMED
    
    # Provenance
    provenance: Optional[EvidenceProvenance] = None
    
    # Timestamps
    collected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Raw data (for drill-down)
    raw_value: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert enums to values
        if isinstance(self.evidence_type, EvidenceType):
            result["evidence_type"] = self.evidence_type.value
        if isinstance(self.status, EvidenceStatus):
            result["status"] = self.status.value
        if self.metric:
            result["metric"] = self.metric.to_dict()
        if self.baseline:
            result["baseline"] = self.baseline.to_dict()
        if self.deviation:
            result["deviation"] = self.deviation.to_dict()
        if self.provenance:
            result["provenance"] = self.provenance.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceCard":
        """Create from dictionary."""
        metric_data = data.get("metric")
        metric = MetricValue.from_dict(metric_data) if metric_data else None
        
        baseline_data = data.get("baseline")
        baseline = Baseline.from_dict(baseline_data) if baseline_data else None
        
        deviation_data = data.get("deviation")
        deviation = Deviation.from_dict(deviation_data) if deviation_data else None
        
        provenance_data = data.get("provenance")
        provenance = None
        if provenance_data:
            source_val = provenance_data.get("source")
            source = EvidenceSource(source_val) if isinstance(source_val, str) else source_val
            provenance = EvidenceProvenance.from_source(source, provenance_data.get("timestamp", ""))
        
        evidence_type = data.get("evidence_type", "metric")
        if isinstance(evidence_type, str):
            try:
                evidence_type = EvidenceType(evidence_type)
            except (ValueError, KeyError):
                evidence_type = EvidenceType.METRIC
        
        status = data.get("status", "confirmed")
        if isinstance(status, str):
            try:
                status = EvidenceStatus(status)
            except (ValueError, KeyError):
                status = EvidenceStatus.CONFIRMED
        
        return cls(
            evidence_id=data.get("evidence_id", f"ev_{uuid.uuid4().hex[:8]}"),
            evidence_type=evidence_type,
            title=data.get("title", ""),
            description=data.get("description", ""),
            metric=metric,
            baseline=baseline,
            deviation=deviation,
            context=data.get("context", {}),
            related_evidence_ids=data.get("related_evidence_ids", []),
            confidence=data.get("confidence", 0.5),
            status=status,
            provenance=provenance,
            collected_at=data.get("collected_at", datetime.utcnow().isoformat()),
            raw_value=data.get("raw_value"),
        )
    
    @classmethod
    def from_metric(
        cls,
        metric_name: str,
        current_value: Union[int, float],
        evidence_type: EvidenceType,
        source: EvidenceSource,
        baseline_value: Optional[Union[int, float]] = None,
        unit: str = "",
        description: str = "",
        confidence: float = 0.5,
        raw_value: Optional[Dict[str, Any]] = None,
        timestamp: str = "",
    ) -> "EvidenceCard":
        """Create an EvidenceCard from a metric with automatic deviation calculation."""
        # Create metric
        metric = MetricValue(
            name=metric_name,
            value=current_value,
            unit=unit,
        )
        
        # Create baseline if provided
        baseline = None
        deviation = None
        if baseline_value is not None:
            baseline = Baseline(value=baseline_value)
            deviation = Deviation.calculate(current_value, baseline_value)
        
        # Create provenance
        provenance = EvidenceProvenance.from_source(source, timestamp)
        
        return cls(
            evidence_id=f"ev_{uuid.uuid4().hex[:8]}",
            evidence_type=evidence_type,
            title=metric_name,
            description=description,
            metric=metric,
            baseline=baseline,
            deviation=deviation,
            confidence=confidence,
            provenance=provenance,
            raw_value=raw_value,
            collected_at=timestamp or datetime.utcnow().isoformat(),
        )
    
    def get_confidence_badge(self) -> Dict[str, Any]:
        """Get confidence badge information."""
        if self.confidence >= 0.95:
            return {"text": f"{self.confidence:.0%}", "label": "High", "color": "green", "icon": "✅"}
        elif self.confidence >= 0.80:
            return {"text": f"{self.confidence:.0%}", "label": "Good", "color": "blue", "icon": "✅"}
        elif self.confidence >= 0.60:
            return {"text": f"{self.confidence:.0%}", "label": "Moderate", "color": "yellow", "icon": "⚠️"}
        elif self.confidence >= 0.40:
            return {"text": f"{self.confidence:.0%}", "label": "Low", "color": "orange", "icon": "⚠️"}
        else:
            return {"text": f"{self.confidence:.0%}", "label": "Very Low", "color": "red", "icon": "❌"}
    
    def get_deviation_display(self) -> Dict[str, Any]:
        """Get formatted deviation display."""
        if not self.deviation:
            return {
                "current": str(self.metric.value) if self.metric else "N/A",
                "baseline": "No baseline",
                "percent": "N/A",
                "severity": "unknown",
                "icon": "❓",
                "color": "gray",
            }
        
        severity = self.deviation.severity
        return {
            "current": f"{self.deviation.current_value:,.2f}" if isinstance(self.deviation.current_value, (int, float)) else str(self.deviation.current_value),
            "baseline": f"{self.deviation.baseline_value:,.2f}" if isinstance(self.deviation.baseline_value, (int, float)) else str(self.deviation.baseline_value),
            "percent": self.deviation.formatted_percent,
            "severity": severity.value,
            "icon": severity.icon,
            "color": severity.color,
            "is_increase_bad": self.deviation.is_increase_bad,
        }


@dataclass
class EvidenceGroup:
    """Group of evidence cards by type or category."""
    group_type: str                         # e.g., "METRIC", "FACT"
    display_name: str = ""                  # Human-readable name
    icon: str = "📊"                        # Icon for the group
    evidence: List[EvidenceCard] = field(default_factory=list)
    count: int = 0
    collapsed: bool = False                  # Default collapsed state
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_type": self.group_type,
            "display_name": self.display_name,
            "icon": self.icon,
            "evidence": [e.to_dict() for e in self.evidence],
            "count": len(self.evidence),
            "collapsed": self.collapsed,
        }


@dataclass
class Contradiction:
    """Represents conflicting evidence."""
    contradiction_id: str = field(default_factory=lambda: f"contra_{uuid.uuid4().hex[:8]}")
    claim: str = ""                         # The claim being contradicted
    supporting_evidence: List[str] = field(default_factory=list)  # Evidence IDs
    contradicting_evidence: List[str] = field(default_factory=list)  # Evidence IDs
    explanation: str = ""                   # Why it's not actually a contradiction
    resolution_status: str = "unresolved"   # "resolved", "unresolved", "pending"
    uncertainty_impact: float = 0.0         # Confidence penalty
    related_question_ids: List[str] = field(default_factory=list)  # Questions to resolve
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MissingEvidence:
    """Represents missing evidence that could improve confidence."""
    missing_id: str = field(default_factory=lambda: f"miss_{uuid.uuid4().hex[:8]}")
    evidence_type: str = ""                 # Type of missing evidence
    description: str = ""                   # Description of what's missing
    required_for: List[str] = field(default_factory=list)  # Root causes that need this
    confidence_penalty: float = 0.0         # Penalty for missing this
    collection_action: Optional[str] = None # SQL or action to collect this
    collection_risk: str = "low"            # "low", "medium", "high"
    is_required: bool = True                # Required vs optional
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceCollection:
    """
    Complete collection of evidence for an incident or analysis.
    
    Contains all evidence cards, groups, contradictions, and missing evidence.
    """
    collection_id: str = field(default_factory=lambda: f"ec_{uuid.uuid4().hex[:8]}")
    incident_id: str = ""
    
    # Evidence
    evidence: List[EvidenceCard] = field(default_factory=list)
    groups: List[EvidenceGroup] = field(default_factory=list)
    
    # Analysis
    contradictions: List[Contradiction] = field(default_factory=list)
    missing_evidence: List[MissingEvidence] = field(default_factory=list)
    
    # Confidence
    overall_confidence: float = 0.5
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    evidence_count: int = 0
    source_counts: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "collection_id": self.collection_id,
            "incident_id": self.incident_id,
            "evidence": [e.to_dict() for e in self.evidence],
            "groups": [g.to_dict() for g in self.groups],
            "contradictions": [c.to_dict() for c in self.contradictions],
            "missing_evidence": [m.to_dict() for m in self.missing_evidence],
            "overall_confidence": self.overall_confidence,
            "confidence_breakdown": self.confidence_breakdown,
            "generated_at": self.generated_at,
            "evidence_count": len(self.evidence),
            "source_counts": self.source_counts,
        }
    
    def group_evidence_by_type(self) -> List[EvidenceGroup]:
        """Group evidence cards by their evidence type."""
        groups: Dict[str, EvidenceGroup] = {}
        
        for card in self.evidence:
            group_type = card.evidence_type.value
            if group_type not in groups:
                evidence_type = card.evidence_type
                groups[group_type] = EvidenceGroup(
                    group_type=group_type,
                    display_name=evidence_type.display_name,
                    icon=evidence_type.icon,
                    evidence=[],
                    collapsed=group_type not in ["METRIC"],  # Default expand metrics
                )
            groups[group_type].evidence.append(card)
            groups[group_type].count += 1
        
        self.groups = list(groups.values())
        return self.groups
    
    def calculate_source_counts(self) -> Dict[str, int]:
        """Count evidence by source."""
        counts: Dict[str, int] = {}
        for card in self.evidence:
            if card.provenance:
                source_name = card.provenance.source_name
                counts[source_name] = counts.get(source_name, 0) + 1
        self.source_counts = counts
        return counts
    
    def get_evidence_by_id(self, evidence_id: str) -> Optional[EvidenceCard]:
        """Get a specific evidence card by ID."""
        for card in self.evidence:
            if card.evidence_id == evidence_id:
                return card
        return None
    
    def get_related_evidence(self, evidence_id: str) -> List[EvidenceCard]:
        """Get all evidence related to a given evidence ID."""
        card = self.get_evidence_by_id(evidence_id)
        if not card:
            return []
        
        related = []
        for related_id in card.related_evidence_ids:
            related_card = self.get_evidence_by_id(related_id)
            if related_card:
                related.append(related_card)
        return related


# =====================================================================
# EVIDENCE CARD FACTORY
# =====================================================================

class EvidenceCardFactory:
    """
    Factory for creating evidence cards from various data sources.
    """
    
    @staticmethod
    def from_pg_stat_statements(data: Dict[str, Any], query: str = "") -> EvidenceCard:
        """Create an EvidenceCard from pg_stat_statements data."""
        return EvidenceCard.from_metric(
            metric_name="query_latency_ms",
            current_value=data.get("mean_time_ms", 0),
            evidence_type=EvidenceType.METRIC,
            source=EvidenceSource.PG_STAT_STATEMENTS,
            baseline_value=100,  # 100ms baseline
            unit="ms",
            description=f"Query: {query[:100]}..." if query else "Query latency",
            confidence=0.9,
            raw_value=data,
        )
    
    @staticmethod
    def from_pg_stat_user_tables(data: Dict[str, Any], table_name: str = "") -> EvidenceCard:
        """Create an EvidenceCard from pg_stat_user_tables data."""
        seq_scans = data.get("seq_scan", 0)
        idx_scans = data.get("idx_scan", 0)
        total_scans = seq_scans + idx_scans
        seq_ratio = seq_scans / total_scans if total_scans > 0 else 0
        
        # Determine severity based on sequential scan ratio
        is_anomaly = seq_ratio > 0.9  # >90% sequential is bad
        
        return EvidenceCard.from_metric(
            metric_name="seq_scan_ratio",
            current_value=seq_ratio * 100,
            evidence_type=EvidenceType.TABLE_STATS,
            source=EvidenceSource.PG_STAT_USER_TABLES,
            baseline_value=10,  # 10% baseline
            unit="%",
            description=f"Table {table_name}: Sequential scan ratio",
            confidence=0.95 if is_anomaly else 0.5,
            raw_value=data,
        )
    
    @staticmethod
    def from_lock_data(blocked_pid: int, blocking_pid: int, wait_duration: float) -> EvidenceCard:
        """Create an EvidenceCard from lock data."""
        return EvidenceCard(
            evidence_id=f"ev_lock_{blocked_pid}_{blocking_pid}",
            evidence_type=EvidenceType.LOCK,
            title=f"Blocking: {blocking_pid} → {blocked_pid}",
            description=f"Process {blocked_pid} blocked by {blocking_pid} for {wait_duration}s",
            metric=MetricValue(
                name="wait_duration_seconds",
                value=wait_duration,
                unit="seconds",
                threshold=60,
                is_anomaly=wait_duration > 60,
            ),
            confidence=0.95,
            status=EvidenceStatus.CONFIRMED,
            provenance=EvidenceProvenance.from_source(
                EvidenceSource.PG_LOCKS,
                datetime.utcnow().isoformat()
            ),
            context={
                "blocked_pid": blocked_pid,
                "blocking_pid": blocking_pid,
            },
        )
    
    @staticmethod
    def from_config_value(
        param_name: str,
        current_value: Union[int, str],
        recommended_value: Union[int, str],
        unit: str = ""
    ) -> EvidenceCard:
        """Create an EvidenceCard from configuration parameter."""
        return EvidenceCard.from_metric(
            metric_name=param_name,
            current_value=current_value,
            evidence_type=EvidenceType.CONFIG,
            source=EvidenceSource.PG_SETTINGS,
            baseline_value=recommended_value,
            unit=unit,
            description=f"Configuration: {param_name}",
            confidence=0.85,
        )
    
    @staticmethod
    def from_explain_analyze(
        plan: Dict[str, Any],
        query: str = ""
    ) -> EvidenceCard:
        """Create an EvidenceCard from EXPLAIN ANALYZE output."""
        node_type = plan.get("Node Type", "Unknown")
        execution_time = plan.get("Actual Total Time", 0)
        
        return EvidenceCard(
            evidence_id=f"ev_plan_{uuid.uuid4().hex[:8]}",
            evidence_type=EvidenceType.EXECUTION_PLAN,
            title=f"Execution Plan: {node_type}",
            description=f"Query plan shows {node_type}",
            metric=MetricValue(
                name="execution_time_ms",
                value=execution_time,
                unit="ms",
            ),
            confidence=0.95,
            status=EvidenceStatus.CONFIRMED,
            provenance=EvidenceProvenance.from_source(
                EvidenceSource.EXPLAIN_ANALYZE,
                datetime.utcnow().isoformat()
            ),
            context={
                "plan": plan,
                "query": query,
            },
            raw_value=plan,
        )


# =====================================================================
# EXPORTS
# =====================================================================

__all__ = [
    "MetricValue",
    "Baseline",
    "Deviation",
    "EvidenceProvenance",
    "EvidenceCard",
    "EvidenceGroup",
    "Contradiction",
    "MissingEvidence",
    "EvidenceCollection",
    "EvidenceCardFactory",
]

