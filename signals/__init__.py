# Package initialization
"""
Signals Module - PostgreSQL Agent Signal Processing

Provides signal detection, evidence building, and root cause analysis
for PostgreSQL performance incidents.

IMPORTANT: This module re-exports classes for convenience.
For production code, import directly from the source modules.
"""

# =====================================================================
# Evidence Types - Re-export from signals.evidence_types
# =====================================================================
from signals.evidence_types import (
    EvidenceSource,
    EvidenceType,
    DeviationSeverity,
    EvidenceStatus,
    get_source_by_name,
    get_evidence_type_by_name,
    calculate_deviation_severity,
)

# =====================================================================
# Evidence Cards - Display-focused models
# =====================================================================
from signals.evidence_card import (
    MetricValue,
    Baseline,
    Deviation,
    EvidenceProvenance,
    EvidenceCard,
    EvidenceGroup,
    Contradiction,
    MissingEvidence,
    EvidenceCollection,
    EvidenceCardFactory,
)

# =====================================================================
# Evidence Explorer
# =====================================================================
from signals.evidence_explorer import (
    EvidenceExplorer,
    create_evidence_explorer,
    build_evidence_from_incident,
)

# =====================================================================
# Core Signal Engine
# =====================================================================
from signals.signal_engine import (
    Signal,
    SignalResult,
    SignalEngine,
)

# =====================================================================
# Evidence Builder
# =====================================================================
from signals.evidence_builder import (
    Evidence,
    EvidenceCollection as OldEvidenceCollection,
    EvidenceBuilder,
)

# =====================================================================
# Root Cause Analysis
# =====================================================================
from signals.root_cause_engine import (
    RootCauseEngine,
    RootCauseResult,
)

# =====================================================================
# Confidence Scoring
# =====================================================================
from signals.confidence_scorer import (
    ConfidenceScorer,
    ConfidenceBreakdown,
)

# =====================================================================
# Clarification Engine
# =====================================================================
from signals.clarification_engine import (
    ClarificationManager,
    ClarificationQuestion,
    ClarificationState,
    QuestionTemplate,
    EvidenceGap,
    EvidenceGapAnalyzer,
)

# =====================================================================
# __all__ - Unified exports
# =====================================================================
__all__ = [
    # Evidence Types (from signals.evidence_types)
    "EvidenceSource",
    "EvidenceType",
    "DeviationSeverity",
    "EvidenceStatus",
    "get_source_by_name",
    "get_evidence_type_by_name",
    "calculate_deviation_severity",
    
    # Evidence Cards (from signals.evidence_card)
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
    
    # Evidence Explorer
    "EvidenceExplorer",
    "create_evidence_explorer",
    "build_evidence_from_incident",
    
    # Signal Engine
    "Signal",
    "SignalResult",
    "SignalEngine",
    
    # Evidence Builder
    "Evidence",
    "OldEvidenceCollection",
    "EvidenceBuilder",
    
    # Root Cause
    "RootCauseEngine",
    "RootCauseResult",
    
    # Confidence
    "ConfidenceScorer",
    "ConfidenceBreakdown",
    
    # Clarification
    "ClarificationManager",
    "ClarificationQuestion",
    "ClarificationState",
    "QuestionTemplate",
    "EvidenceGap",
    "EvidenceGapAnalyzer",
]

