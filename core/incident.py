"""
Core Incident Model - Unified incident object that flows through the pipeline.

This is the central data structure that carries all incident information
from signal generation through analysis to output. All pipeline stages
contribute to building a complete Incident object.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.data_models import (
    Evidence,
    Confidence,
    Context,
    ImpactMetrics,
    Action,
    RootCauseDetail,
    ClarificationQuestion,
)
from core.enums import Severity, Category


@dataclass
class IncidentMetadata:
    """Metadata about the incident detection and analysis."""
    incident_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Analysis info
    detection_source: str = ""         # e.g., "signal_engine", "kb_matcher", "user_report"
    analysis_duration_ms: float = 0.0
    generator_version: str = "pg-agent-v1.0.0"
    
    # Status
    is_resolved: bool = False
    resolution_actions_taken: List[str] = field(default_factory=list)
    
    # Tags for categorization
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class Incident:
    """Unified incident object - the primary data carrier through the pipeline.
    
    This object is built incrementally:
    1. SignalEngine.process() → sets signals, initial summary
    2. EvidenceBuilder.build() → sets evidence
    3. RootCauseEngine.analyze() → sets root_causes
    4. ConfidenceScorer.calculate() → sets confidence
    5. PgRecommender.recommend() → sets recommended_actions
    6. ClarificationEngine.generate() → sets clarification_questions
    7. ImpactAnalyzer.analyze() → sets impact
    
    Then output renderers transform this to user-facing formats.
    """
    
    # ===== IDENTITY =====
    metadata: IncidentMetadata = field(default_factory=IncidentMetadata)
    
    # ===== SUMMARY =====
    summary: str = ""
    description: str = ""
    severity: Severity = Severity.HIGH
    category: Category = Category.QUERY_PERFORMANCE
    
    # ===== CONTEXT =====
    context: Context = field(default_factory=Context)
    
    # ===== ROOT CAUSES =====
    root_causes: List[RootCauseDetail] = field(default_factory=list)
    likely_root_cause_index: int = 0   # Index into root_causes of most likely
    
    # ===== EVIDENCE =====
    evidence: List[Evidence] = field(default_factory=list)
    evidence_summary: Dict[str, int] = field(default_factory=dict)  # Type → count
    
    # ===== CONFIDENCE =====
    confidence: Confidence = field(default_factory=Confidence)
    
    # ===== IMPACT =====
    impact: ImpactMetrics = field(default_factory=ImpactMetrics)
    affected_tables: List[str] = field(default_factory=list)
    affected_queries: List[str] = field(default_factory=list)
    
    # ===== RECOMMENDATIONS =====
    recommended_actions: List[Action] = field(default_factory=list)
    immediate_actions: List[Action] = field(default_factory=list)
    long_term_fixes: List[Action] = field(default_factory=list)
    validation_steps: List[str] = field(default_factory=list)
    preventive_actions: List[str] = field(default_factory=list)
    
    # ===== CLARIFICATION =====
    clarification_questions: List[ClarificationQuestion] = field(default_factory=list)
    clarifications_requested: bool = False
    
    # ===== KNOWLEDGE BASE REFERENCES =====
    kb_entry_ids: List[str] = field(default_factory=list)
    similar_incidents: List[Dict[str, Any]] = field(default_factory=list)
    
    # ===== EXTENSIBILITY =====
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    # ===== METHODS =====

    def to_dict(self) -> Dict[str, Any]:
        """Convert incident to dictionary for serialization."""
        from dataclasses import asdict
        result = {
            "metadata": self.metadata.to_dict(),
            "summary": self.summary,
            "description": self.description,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context.to_dict(),
            "root_causes": [rc.to_dict() for rc in self.root_causes],
            "likely_root_cause_index": self.likely_root_cause_index,
            "evidence": [e.to_dict() for e in self.evidence],
            "evidence_summary": self.evidence_summary,
            "confidence": self.confidence.to_dict(),
            "impact": self.impact.to_dict(),
            "affected_tables": self.affected_tables,
            "affected_queries": self.affected_queries,
            "recommended_actions": [a.to_dict() for a in self.recommended_actions],
            "immediate_actions": [a.to_dict() for a in self.immediate_actions],
            "long_term_fixes": [a.to_dict() for a in self.long_term_fixes],
            "validation_steps": self.validation_steps,
            "preventive_actions": self.preventive_actions,
            "clarification_questions": [q.to_dict() for q in self.clarification_questions],
            "clarifications_requested": self.clarifications_requested,
            "kb_entry_ids": self.kb_entry_ids,
            "similar_incidents": self.similar_incidents,
            "custom_fields": self.custom_fields,
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Incident":
        """Create incident from dictionary."""
        from core.data_models import RootCauseDetail, ClarificationQuestion
        
        return cls(
            metadata=IncidentMetadata(**data.get("metadata", {})),
            summary=data.get("summary", ""),
            description=data.get("description", ""),
            severity=Severity(data.get("severity", "high")),
            category=Category(data.get("category", "query_performance")),
            context=Context.from_dict(data.get("context", {})),
            root_causes=[RootCauseDetail.from_dict(rc) for rc in data.get("root_causes", [])],
            likely_root_cause_index=data.get("likely_root_cause_index", 0),
            evidence=[Evidence.from_dict(e) for e in data.get("evidence", [])],
            evidence_summary=data.get("evidence_summary", {}),
            confidence=Confidence(**data.get("confidence", {})),
            impact=ImpactMetrics.from_dict(data.get("impact", {})),
            affected_tables=data.get("affected_tables", []),
            affected_queries=data.get("affected_queries", []),
            recommended_actions=[Action.from_dict(a) for a in data.get("recommended_actions", [])],
            immediate_actions=[Action.from_dict(a) for a in data.get("immediate_actions", [])],
            long_term_fixes=[Action.from_dict(a) for a in data.get("long_term_fixes", [])],
            validation_steps=data.get("validation_steps", []),
            preventive_actions=data.get("preventive_actions", []),
            clarification_questions=[ClarificationQuestion.from_dict(q) for q in data.get("clarification_questions", [])],
            clarifications_requested=data.get("clarifications_requested", False),
            kb_entry_ids=data.get("kb_entry_ids", []),
            similar_incidents=data.get("similar_incidents", []),
            custom_fields=data.get("custom_fields", {}),
        )

    def get_likely_root_cause(self) -> Optional[RootCauseDetail]:
        """Get the most likely root cause."""
        if 0 <= self.likely_root_cause_index < len(self.root_causes):
            return self.root_causes[self.likely_root_cause_index]
        return self.root_causes[0] if self.root_causes else None

    def needs_clarification(self, confidence_threshold: float = 0.75) -> bool:
        """Check if incident needs clarification to reach confidence threshold."""
        return self.confidence.overall_score < confidence_threshold

    def actions_requiring_approval(self) -> List[Action]:
        """Get all actions that require approval."""
        all_actions = self.recommended_actions + self.immediate_actions + self.long_term_fixes
        return [a for a in all_actions if a.requires_approval]

    def actions_by_risk(self) -> Dict[str, List[Action]]:
        """Get actions grouped by risk level."""
        all_actions = self.recommended_actions + self.immediate_actions + self.long_term_fixes
        by_risk = {}
        for action in all_actions:
            risk_key = action.risk.value
            if risk_key not in by_risk:
                by_risk[risk_key] = []
            by_risk[risk_key].append(action)
        return by_risk

    def confidence_breakdown_summary(self) -> Dict[str, float]:
        """Get confidence breakdown as dict."""
        return {
            "overall": self.confidence.overall_score,
            "base": self.confidence.base_confidence,
            "completeness": self.confidence.evidence_completeness,
            "agreement": self.confidence.signal_agreement,
            "freshness": self.confidence.data_freshness,
            "conflict_penalty": self.confidence.conflict_penalty,
        }

    def is_critical(self) -> bool:
        """Check if incident is critical."""
        return self.severity == Severity.CRITICAL

    def is_resolved(self) -> bool:
        """Check if incident is resolved."""
        return self.metadata.is_resolved

    def mark_resolved(self, actions_taken: List[str]) -> None:
        """Mark incident as resolved."""
        self.metadata.is_resolved = True
        self.metadata.resolution_actions_taken = actions_taken
        self.metadata.updated_at = datetime.utcnow().isoformat()
