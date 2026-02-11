"""
Core Enums - Single source of truth for all enumeration types.

This module centralizes all enum definitions to prevent duplication
and ensure consistency across the system.
"""

from enum import Enum


# =====================================================================
# SEVERITY & CATEGORIZATION
# =====================================================================

class Severity(Enum):
    """Issue severity levels - standardized across KB and incidents."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Category(Enum):
    """Problem categories - standardized across KB and incident types."""
    QUERY_PERFORMANCE = "query_performance"
    INDEX_HEALTH = "index_health"
    LOCKING = "locking"
    MAINTENANCE = "maintenance"
    CONFIGURATION = "configuration"
    HARDWARE_CAPACITY = "hardware_capacity"
    APPLICATION_IMPACT = "application_impact"
    INCIDENT_RESPONSE = "incident_response"


class Component(Enum):
    """PostgreSQL components that can be affected."""
    PLANNER = "planner"
    EXECUTOR = "executor"
    IO = "io"
    MEMORY = "memory"
    CPU = "cpu"
    STORAGE = "storage"
    LOCK_MANAGER = "lock_manager"
    VACUUM = "vacuum"
    ANALYZE = "analyze"
    BUFFER_POOL = "buffer_pool"
    INDEX_MANAGER = "index_manager"
    TRANSACTION_MANAGER = "transaction_manager"


# =====================================================================
# ROOT CAUSE ANALYSIS
# =====================================================================

class RootCauseCategory(Enum):
    """Root cause categories for systematic analysis.
    
    Note: Values are lowercase to match signal naming conventions.
    """
    INDEX_ISSUES = "INDEX_ISSUES"
    STATISTICS_MAINTENANCE = "STATISTICS_MAINTENANCE"
    BLOCKING_LOCKING = "BLOCKING_LOCKING"
    CONFIGURATION = "CONFIGURATION"
    APPLICATION_BEHAVIOR = "APPLICATION_BEHAVIOR"
    CAPACITY_HARDWARE = "CAPACITY_HARDWARE"
    BACKGROUND_JOBS = "BACKGROUND_JOBS"
    DEPLOYMENT_SCHEMA = "DEPLOYMENT_SCHEMA"
    
    @classmethod
    def from_string(cls, value: str) -> "RootCauseCategory":
        """Convert string to enum, supporting both cases."""
        value_upper = value.upper()
        value_lower = value.lower()
        for member in cls:
            if member.value == value_upper or member.value == value_lower:
                return member
        raise ValueError(f"Unknown RootCauseCategory: {value}")


# =====================================================================
# EVIDENCE & SIGNALS
# =====================================================================

class EvidenceType(Enum):
    """Types of evidence that support root cause analysis."""
    QUERY_METRICS = "query_metrics"
    TABLE_STATISTICS = "table_statistics"
    INDEX_HEALTH = "index_health"
    LOCKING = "locking"
    CONFIGURATION = "configuration"
    HARDWARE_CAPACITY = "hardware_capacity"
    BACKGROUND_JOBS = "background_jobs"
    DEPLOYMENT_SCHEMA = "deployment_schema"
    MAINTENANCE = "maintenance"
    FACT = "fact"
    EXPLANATION = "explanation"


# =====================================================================
# CLARIFICATION QUESTIONS
# =====================================================================

class QuestionCategory(Enum):
    """Categories of clarification questions."""
    DEPLOYMENT = "deployment"
    PERFORMANCE_BASELINE = "performance_baseline"
    DATA_OPERATIONS = "data_operations"
    MAINTENANCE = "maintenance"
    CONFIGURATION = "configuration"


class QuestionPriority(Enum):
    """Priority levels for clarification questions."""
    CRITICAL = "critical"  # Must answer to proceed
    HIGH = "high"  # Significantly improves confidence
    MEDIUM = "medium"  # Moderately helpful
    LOW = "low"  # Nice to have


# =====================================================================
# ACTIONS & RECOMMENDATIONS
# =====================================================================

class ActionRisk(Enum):
    """Risk level for recommended actions."""
    LOW = "low"              # Safe, reversible, no locks
    MEDIUM = "medium"        # Some risk, requires care
    HIGH = "high"            # Significant risk, may need approval
    CRITICAL = "critical"    # Very risky, must have approval


class ActionApprovalStatus(Enum):
    """Approval status for actions requiring authorization."""
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


# =====================================================================
# OUTPUT FORMATS
# =====================================================================

class OutputFormat(Enum):
    """Supported output formats for incident rendering."""
    TERMINAL = "terminal"    # ANSI-colored console output
    MARKDOWN = "markdown"     # GitHub-flavored markdown
    SLACK = "slack"          # Slack message format
    EMAIL = "email"          # HTML email format
    JSON = "json"            # Structured JSON
    HTML = "html"            # HTML report


# =====================================================================
# SIGNAL TYPES
# =====================================================================

class SignalType(Enum):
    """Types of signals from signal engine."""
    INCIDENT = "incident"
    QUERY_METRICS = "query_metrics"
    TABLE_STATS = "table_stats"
    INDEX_HEALTH = "index_health"
    LOCKING = "locking"
    CONFIGURATION = "configuration"
    HARDWARE = "hardware"
    SQL = "sql"
    TEXT = "text"
    PGBENCH = "pgbench"
    SECURITY = "security"
    VACUUM = "vacuum"


# =====================================================================
# SECURITY ENUMS
# =====================================================================

class SecurityCategory(Enum):
    """Security issue categories for PostgreSQL."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ENCRYPTION = "encryption"
    AUDIT_LOGGING = "audit_logging"
    NETWORK_SECURITY = "network_security"
    DATA_PROTECTION = "data_protection"
    ACCESS_CONTROL = "access_control"
    COMPLIANCE = "compliance"


class SecuritySeverity(Enum):
    """Security issue severity levels."""
    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"          # Address within 24 hours
    MEDIUM = "medium"      # Address within 1 week
    LOW = "low"            # Address during next maintenance window
    INFO = "info"          # Informational


# =====================================================================
# VACUUM/MAINTENANCE ENUMS
# =====================================================================

class VacuumMetricType(Enum):
    """Types of VACUUM/maintenance metrics."""
    DEAD_TUPLE_RATIO = "dead_tuple_ratio"
    TABLE_BLOAT = "table_bloat"
    INDEX_BLOAT = "index_bloat"
    LAST_VACUUM_AGE = "last_vacuum_age"
    LAST_ANALYZE_AGE = "last_analyze_age"
    AUTOVACUUM_THRESHOLD = "autovacuum_threshold"
    VACUUM_PROGRESS = "vacuum_progress"


class MaintenanceStatus(Enum):
    """Maintenance operation status."""
    HEALTHY = "healthy"
    NEEDS_ATTENTION = "needs_attention"
    CRITICAL = "critical"
    MAINTENANCE_REQUIRED = "maintenance_required"
