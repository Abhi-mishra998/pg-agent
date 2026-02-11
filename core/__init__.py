"""
Core Package - Unified Data Models and Enums

This package provides the single source of truth for all data models
used across the pg-agent system. All modules should import from here
rather than defining their own classes.

Structure:
  - enums.py: All enum types (Severity, Category, EvidenceType, etc.)
  - data_models.py: All dataclass definitions (Action, Evidence, Confidence, etc.)
  - incident.py: Unified Incident object
  - pipeline.py: Pipeline interfaces and protocols
"""

# Core Enums - Import first to avoid circular dependencies
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
    SignalType,
    OutputFormat,
)

# Core Data Models
from core.data_models import (
    MetricValue,
    Action,
    Evidence,
    Confidence,
    Context,
    ImpactMetrics,
    DatabaseEnvironment,
    QueryFingerprint,
    RootCauseDetail,
    ClarificationQuestion,
)

# Unified Incident
from core.incident import (
    Incident,
    IncidentMetadata,
)

# Pipeline Orchestration
from core.pipeline import (
    Pipeline,
    PipelineResult,
    PipelineMetrics,
    SignalAnalyzer,
    EvidenceCollector,
    RecommenderEngine,
    OutputFormatter,
    create_default_pipeline,
    create_lightweight_pipeline,
)

__all__ = [
    # Enums
    "Severity",
    "Category",
    "Component",
    "RootCauseCategory",
    "EvidenceType",
    "QuestionCategory",
    "QuestionPriority",
    "ActionRisk",
    "ActionApprovalStatus",
    "SignalType",
    "OutputFormat",
    # Data Models
    "MetricValue",
    "Action",
    "Evidence",
    "Confidence",
    "Context",
    "ImpactMetrics",
    "DatabaseEnvironment",
    "QueryFingerprint",
    "RootCauseDetail",
    "ClarificationQuestion",
    # Unified Objects
    "Incident",
    "IncidentMetadata",
    # Pipeline
    "Pipeline",
    "PipelineResult",
    "PipelineMetrics",
    "SignalAnalyzer",
    "EvidenceCollector",
    "RecommenderEngine",
    "OutputFormatter",
    "create_default_pipeline",
    "create_lightweight_pipeline",
]
