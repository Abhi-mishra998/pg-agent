# Recommendations Package

# PostgreSQL Knowledge Base and Recommender System
from .kb_schema import (
    KBEntry,
    KBVersion,
    Metadata,
    ProblemIdentity,
    DetectionSignals,
    RootCauseAnalysis,
    ImpactAnalysis,
    Recommendations,
    Action,
    Evidence,
    Confidence,
    Context,
    QueryFingerprint,
    Environment,
    Severity,
    Category,
    Component,
)

from .kb_loader import KBLoader, KBFileWatcher, load_kb_from_default_locations

from .kb_index import KBVectorIndex, KBRuleMatcher, SearchResult

from .pg_recommender import PgRecommender, RecommendationReport, Recommendation

# Review and Approval System
from .review_schema import (
    ReviewCard,
    ReviewAction,
    ApprovalWorkflow,
    ApprovalStep,
    AuditTrail,
    AuditEntry,
    RiskIndicators,
    SafetyWarnings,
    RollbackPlan,
)

from .approval_store import save_card, load_card, append_audit_entry

from .api import app as review_api_app

from .review_renderer import (
    render_to_terminal,
    render_to_markdown,
    render_to_html,
    render_to_json,
    render_session_to_terminal,
)

__all__ = [
    # KB Schema
    "KBEntry",
    "KBVersion",
    "Metadata",
    "ProblemIdentity",
    "DetectionSignals",
    "RootCauseAnalysis",
    "ImpactAnalysis",
    "Recommendations",
    "Action",
    "Evidence",
    "Confidence",
    "Context",
    "QueryFingerprint",
    "Environment",
    "Severity",
    "Category",
    "Component",
    # KB Loader
    "KBLoader",
    "KBFileWatcher",
    "load_kb_from_default_locations",
    # KB Index
    "KBVectorIndex",
    "KBRuleMatcher",
    "SearchResult",
    # Recommender
    "PgRecommender",
    "RecommendationReport",
    "Recommendation",
    # Review and Approval System
    "ReviewCard",
    "ReviewAction",
    "ApprovalWorkflow",
    "ApprovalStep",
    "AuditTrail",
    "AuditEntry",
    "RiskIndicators",
    "SafetyWarnings",
    "RollbackPlan",
    "save_card",
    "load_card",
    "append_audit_entry",
    "review_api_app",
    "render_to_terminal",
    "render_to_markdown",
    "render_to_html",
    "render_to_json",
    "render_session_to_terminal",
]

