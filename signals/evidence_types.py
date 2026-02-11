#!/usr/bin/env python3
"""
Evidence Types - Enums and Constants for Evidence Sources

DEPRECATION NOTICE:
This module is being phased out. Please use:
- EvidenceType from core.enums
- EvidenceSource from core.enums

The canonical definitions are now in core/enums.py.
This module is kept for backward compatibility only.
"""

# Import from core.enums for consistency
# The canonical EvidenceType enum is in core/enums.py
try:
    from core.enums import EvidenceType as CoreEvidenceType
    from core.enums import EvidenceSource as CoreEvidenceSource
    from core.enums import DeviationSeverity as CoreDeviationSeverity
    from core.enums import EvidenceStatus as CoreEvidenceStatus
    
    # Use core enums as canonical
    EvidenceType = CoreEvidenceType
    EvidenceSource = CoreEvidenceSource
    DeviationSeverity = CoreDeviationSeverity
    EvidenceStatus = CoreEvidenceStatus
    
except ImportError:
    # Fallback to local definitions if core.enums doesn't have them yet
    # This is a temporary bridge until core.enums is fully populated
    from enum import Enum
    from typing import Dict, List, Optional
    from dataclasses import dataclass


# =====================================================================
# EVIDENCE SOURCE ENUMS (Fallback - kept for backward compatibility)
# =====================================================================

class EvidenceSource(Enum):
    """
    PostgreSQL and system sources for evidence collection.
    
    Each source has associated metadata for UI rendering.
    """
    
    # PostgreSQL Statistics
    PG_STAT_STATEMENTS = "pg_stat_statements"
    PG_STAT_USER_TABLES = "pg_stat_user_tables"
    PG_STAT_USER_INDEXES = "pg_stat_user_indexes"
    PG_STATIO_USER_TABLES = "pg_statio_user_tables"
    PG_STAT_ACTIVITY = "pg_stat_activity"
    PG_LOCKS = "pg_locks"
    PG_SETTINGS = "pg_settings"
    PG_INDEXES = "pg_indexes"
    PG_CLASS = "pg_class"
    
    # Execution Plans
    EXPLAIN_ANALYZE = "EXPLAIN ANALYZE"
    
    # Application Signals
    APPLICATION_LOGS = "application_logs"
    APM_METRICS = "apm_metrics"
    
    # System Metrics
    SYSTEM_CPU = "system_cpu"
    SYSTEM_MEMORY = "system_memory"
    SYSTEM_IO = "system_io"
    SYSTEM_DISK = "system_disk"
    
    # Knowledge Base
    KNOWLEDGE_BASE = "knowledge_base"
    BEST_PRACTICES = "best_practices"
    
    @property
    def icon(self) -> str:
        """Get the icon for this evidence source."""
        icons = {
            EvidenceSource.PG_STAT_STATEMENTS: "📊",
            EvidenceSource.PG_STAT_USER_TABLES: "📋",
            EvidenceSource.PG_STAT_USER_INDEXES: "📈",
            EvidenceSource.PG_STATIO_USER_TABLES: "💾",
            EvidenceSource.PG_STAT_ACTIVITY: "👥",
            EvidenceSource.PG_LOCKS: "🔒",
            EvidenceSource.PG_SETTINGS: "⚙️",
            EvidenceSource.PG_INDEXES: "📑",
            EvidenceSource.PG_CLASS: "📦",
            EvidenceSource.EXPLAIN_ANALYZE: "🗺️",
            EvidenceSource.APPLICATION_LOGS: "📝",
            EvidenceSource.APM_METRICS: "📡",
            EvidenceSource.SYSTEM_CPU: "🖥️",
            EvidenceSource.SYSTEM_MEMORY: "🧠",
            EvidenceSource.SYSTEM_IO: "💿",
            EvidenceSource.SYSTEM_DISK: "💾",
            EvidenceSource.KNOWLEDGE_BASE: "📚",
            EvidenceSource.BEST_PRACTICES: "✅",
        }
        return icons.get(self, "❓")
    
    @property
    def color(self) -> str:
        """Get the UI color class for this source."""
        colors = {
            EvidenceSource.PG_STAT_STATEMENTS: "blue",
            EvidenceSource.PG_STAT_USER_TABLES: "green",
            EvidenceSource.PG_STAT_USER_INDEXES: "green",
            EvidenceSource.PG_STATIO_USER_TABLES: "cyan",
            EvidenceSource.PG_STAT_ACTIVITY: "blue",
            EvidenceSource.PG_LOCKS: "red",
            EvidenceSource.PG_SETTINGS: "purple",
            EvidenceSource.PG_INDEXES: "green",
            EvidenceSource.PG_CLASS: "gray",
            EvidenceSource.EXPLAIN_ANALYZE: "orange",
            EvidenceSource.APPLICATION_LOGS: "brown",
            EvidenceSource.APM_METRICS: "blue",
            EvidenceSource.SYSTEM_CPU: "gray",
            EvidenceSource.SYSTEM_MEMORY: "gray",
            EvidenceSource.SYSTEM_IO: "gray",
            EvidenceSource.SYSTEM_DISK: "gray",
            EvidenceSource.KNOWLEDGE_BASE: "purple",
            EvidenceSource.BEST_PRACTICES: "green",
        }
        return colors.get(self, "gray")
    
    @property
    def display_name(self) -> str:
        """Get the human-readable display name."""
        names = {
            EvidenceSource.PG_STAT_STATEMENTS: "pg_stat_statements",
            EvidenceSource.PG_STAT_USER_TABLES: "pg_stat_user_tables",
            EvidenceSource.PG_STAT_USER_INDEXES: "pg_stat_user_indexes",
            EvidenceSource.PG_STATIO_USER_TABLES: "pg_statio_user_tables",
            EvidenceSource.PG_STAT_ACTIVITY: "pg_stat_activity",
            EvidenceSource.PG_LOCKS: "pg_locks",
            EvidenceSource.PG_SETTINGS: "pg_settings",
            EvidenceSource.PG_INDEXES: "pg_indexes",
            EvidenceSource.PG_CLASS: "pg_class",
            EvidenceSource.EXPLAIN_ANALYZE: "EXPLAIN ANALYZE",
            EvidenceSource.APPLICATION_LOGS: "Application Logs",
            EvidenceSource.APM_METRICS: "APM Metrics",
            EvidenceSource.SYSTEM_CPU: "System CPU",
            EvidenceSource.SYSTEM_MEMORY: "System Memory",
            EvidenceSource.SYSTEM_IO: "System I/O",
            EvidenceSource.SYSTEM_DISK: "System Disk",
            EvidenceSource.KNOWLEDGE_BASE: "Knowledge Base",
            EvidenceSource.BEST_PRACTICES: "Best Practices",
        }
        return names.get(self, self.value)


class EvidenceType(Enum):
    """
    Types of evidence based on content.
    
    DEPRECATED: Use core.enums.EvidenceType instead.
    """
    
    METRIC = "METRIC"
    TABLE_STATS = "TABLE_STATS"
    INDEX_STATS = "INDEX_STATS"
    LOCK = "LOCK"
    CONFIG = "CONFIG"
    QUERY = "QUERY"
    EXECUTION_PLAN = "EXECUTION_PLAN"
    LOG_ENTRY = "LOG_ENTRY"
    FACT = "FACT"
    EXPLANATION = "EXPLANATION"
    SESSION = "SESSION"
    HARDWARE = "HARDWARE"
    
    @property
    def icon(self) -> str:
        """Get the icon for this evidence type."""
        icons = {
            EvidenceType.METRIC: "📊",
            EvidenceType.TABLE_STATS: "📋",
            EvidenceType.INDEX_STATS: "📈",
            EvidenceType.LOCK: "🔒",
            EvidenceType.CONFIG: "⚙️",
            EvidenceType.QUERY: "📝",
            EvidenceType.EXECUTION_PLAN: "🗺️",
            EvidenceType.LOG_ENTRY: "📋",
            EvidenceType.FACT: "✅",
            EvidenceType.EXPLANATION: "💡",
            EvidenceType.SESSION: "👥",
            EvidenceType.HARDWARE: "🖥️",
        }
        return icons.get(self, "❓")
    
    @property
    def display_name(self) -> str:
        """Get the human-readable display name."""
        names = {
            EvidenceType.METRIC: "Metric",
            EvidenceType.TABLE_STATS: "Table Statistics",
            EvidenceType.INDEX_STATS: "Index Statistics",
            EvidenceType.LOCK: "Lock",
            EvidenceType.CONFIG: "Configuration",
            EvidenceType.QUERY: "Query",
            EvidenceType.EXECUTION_PLAN: "Execution Plan",
            EvidenceType.LOG_ENTRY: "Log Entry",
            EvidenceType.FACT: "Fact",
            EvidenceType.EXPLANATION: "Explanation",
            EvidenceType.SESSION: "Session",
            EvidenceType.HARDWARE: "Hardware",
        }
        return names.get(self, self.value)


class DeviationSeverity(Enum):
    """
    Severity level for metric deviation from baseline.
    """
    
    NORMAL = "normal"
    MINOR = "minor"
    SIGNIFICANT = "significant"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    
    @property
    def color(self) -> str:
        colors = {
            DeviationSeverity.NORMAL: "green",
            DeviationSeverity.MINOR: "yellow",
            DeviationSeverity.SIGNIFICANT: "orange",
            DeviationSeverity.CRITICAL: "red",
            DeviationSeverity.UNKNOWN: "gray",
        }
        return colors.get(self, "gray")
    
    @property
    def icon(self) -> str:
        icons = {
            DeviationSeverity.NORMAL: "✅",
            DeviationSeverity.MINOR: "⚠️",
            DeviationSeverity.SIGNIFICANT: "🟠",
            DeviationSeverity.CRITICAL: "🚨",
            DeviationSeverity.UNKNOWN: "❓",
        }
        return icons.get(self, "❓")


class EvidenceStatus(Enum):
    """
    Status of an evidence item.
    """
    
    CONFIRMED = "confirmed"
    PARTIAL = "partial"
    CONFLICTING = "conflicting"
    MISSING = "missing"
    STALE = "stale"
    PENDING = "pending"


# =====================================================================
# EVIDENCE SOURCE MAPPING (kept for backward compatibility)
# =====================================================================

# Map evidence types to their common sources
EVIDENCE_TYPE_TO_SOURCES = {
    EvidenceType.METRIC: [
        EvidenceSource.PG_STAT_STATEMENTS,
        EvidenceSource.PG_STAT_USER_TABLES,
        EvidenceSource.PG_STAT_USER_INDEXES,
        EvidenceSource.SYSTEM_CPU,
        EvidenceSource.SYSTEM_MEMORY,
        EvidenceSource.SYSTEM_IO,
    ],
    EvidenceType.TABLE_STATS: [
        EvidenceSource.PG_STAT_USER_TABLES,
        EvidenceSource.PG_CLASS,
        EvidenceSource.PG_INDEXES,
    ],
    EvidenceType.INDEX_STATS: [
        EvidenceSource.PG_STAT_USER_INDEXES,
        EvidenceSource.PG_CLASS,
        EvidenceSource.PG_INDEXES,
    ],
    EvidenceType.LOCK: [
        EvidenceSource.PG_LOCKS,
        EvidenceSource.PG_STAT_ACTIVITY,
    ],
    EvidenceType.CONFIG: [
        EvidenceSource.PG_SETTINGS,
    ],
    EvidenceType.QUERY: [
        EvidenceSource.PG_STAT_STATEMENTS,
        EvidenceSource.APPLICATION_LOGS,
    ],
    EvidenceType.EXECUTION_PLAN: [
        EvidenceSource.EXPLAIN_ANALYZE,
    ],
    EvidenceType.LOG_ENTRY: [
        EvidenceSource.APPLICATION_LOGS,
    ],
    EvidenceType.FACT: [
        EvidenceSource.PG_INDEXES,
        EvidenceSource.PG_CLASS,
        EvidenceSource.KNOWLEDGE_BASE,
    ],
    EvidenceType.EXPLANATION: [
        EvidenceSource.KNOWLEDGE_BASE,
        EvidenceSource.BEST_PRACTICES,
    ],
    EvidenceType.SESSION: [
        EvidenceSource.PG_STAT_ACTIVITY,
    ],
    EvidenceType.HARDWARE: [
        EvidenceSource.SYSTEM_CPU,
        EvidenceSource.SYSTEM_MEMORY,
        EvidenceSource.SYSTEM_IO,
        EvidenceSource.SYSTEM_DISK,
    ],
}

# Map sources to their evidence types
SOURCE_TO_EVIDENCE_TYPES = {
    source: types for types, sources in EVIDENCE_TYPE_TO_SOURCES.items() 
    for source in sources
}


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def get_source_by_name(name: str) -> Optional[EvidenceSource]:
    """Get EvidenceSource enum by name string."""
    for source in EvidenceSource:
        if source.value == name or source.name == name:
            return source
    return None


def get_evidence_type_by_name(name: str) -> Optional[EvidenceType]:
    """Get EvidenceType enum by name string."""
    for etype in EvidenceType:
        if etype.value == name or etype.name == name:
            return etype
    return None


def calculate_deviation_severity(
    current_value: float, 
    baseline_value: float,
    increase_is_bad: bool = True
) -> DeviationSeverity:
    """Calculate deviation severity from current and baseline values."""
    if baseline_value == 0:
        if current_value == 0:
            return DeviationSeverity.NORMAL
        return DeviationSeverity.CRITICAL
    
    ratio = current_value / baseline_value
    
    if increase_is_bad:
        deviation_percent = (ratio - 1) * 100
    else:
        deviation_percent = (1 - ratio) * 100
    
    if abs(deviation_percent) < 10:
        return DeviationSeverity.NORMAL
    elif abs(deviation_percent) < 100:
        return DeviationSeverity.MINOR
    elif abs(deviation_percent) < 1000:
        return DeviationSeverity.SIGNIFICANT
    else:
        return DeviationSeverity.CRITICAL


def format_deviation(
    current_value: float,
    baseline_value: float,
    unit: str = ""
) -> Dict[str, any]:
    """
    Format deviation information for display.
    
    Returns:
        Dictionary with deviation details
    """
    if baseline_value == 0:
        return {
            "absolute": current_value,
            "percent": "N/A",
            "formatted": f"{current_value:,.2f} {unit}" if unit else f"{current_value:,.2f}",
            "severity": DeviationSeverity.UNKNOWN,
            "message": "No baseline available"
        }
    
    absolute = current_value - baseline_value
    if baseline_value != 0:
        percent = ((current_value - baseline_value) / baseline_value) * 100
    else:
        percent = float('inf') if current_value > 0 else float('-inf')
    
    # Format based on severity
    if percent > 0:
        sign = "+"
    else:
        sign = ""
    
    severity = calculate_deviation_severity(current_value, baseline_value)
    
    return {
        "absolute": absolute,
        "percent": percent,
        "formatted_percent": f"{sign}{percent:,.0f}%",
        "formatted": f"{sign}{percent:,.0f}%",
        "severity": severity,
        "current_formatted": f"{current_value:,.2f} {unit}".strip() if unit else f"{current_value:,.2f}",
        "baseline_formatted": f"{baseline_value:,.2f} {unit}".strip() if unit else f"{baseline_value:,.2f}",
    }


# =====================================================================
# EXPORTS
# =====================================================================

__all__ = [
    # Enums
    "EvidenceSource",
    "EvidenceType",  # Use core.enums.EvidenceType instead
    "DeviationSeverity",
    "EvidenceStatus",
    
    # Mappings (kept for backward compatibility)
    "EVIDENCE_TYPE_TO_SOURCES",
    "SOURCE_TO_EVIDENCE_TYPES",
    
    # Helper functions
    "get_source_by_name",
    "get_evidence_type_by_name",
    "calculate_deviation_severity",
    "format_deviation",
]

