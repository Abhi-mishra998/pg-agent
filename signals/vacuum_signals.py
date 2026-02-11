#!/usr/bin/env python3
"""
VACUUM Signals Module

Generate VACUUM/maintenance signals from PostgreSQL statistics.
Designed for proactive maintenance monitoring and optimization.

VACUUM Categories:
- Dead Tuple Management (ratio monitoring, bloat detection)
- Statistics Freshness (analyze age, autovacuum thresholds)
- Autovacuum Configuration (worker count, cost delay)
- Table/Index Bloat (space reclamation needs)
- Maintenance Window Planning (scheduled maintenance)
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from signals.signal_engine import Signal, SignalResult


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

# VACUUM thresholds (industry-aligned)
DEAD_TUPLE_THRESHOLD = 0.20  # 20% dead tuples
STALE_STATS_THRESHOLD_DAYS = 7  # 7 days since ANALYZE
TABLE_BLOAT_THRESHOLD = 0.30  # 30% table bloat
INDEX_BLOAT_THRESHOLD = 0.30  # 30% index bloat
HIGH_BLOAT_THRESHOLD = 0.50  # 50% bloat - critical

# Autovacuum thresholds
AUTOVACUUM_VACUUM_SCALE_FACTOR = 0.20  # Default PostgreSQL
AUTOVACUUM_ANALYZE_SCALE_FACTOR = 0.10
AUTOVACUUM_NAPTIME = "60s"  # Default
AUTOVACUUM_MAX_WORKERS = 3  # Default


# -------------------------------------------------------------------
# Data Models
# -------------------------------------------------------------------

@dataclass
class VacuumMetric:
    """A single VACUUM metric."""
    metric_name: str
    table_name: Optional[str]
    value: float
    unit: str
    threshold: float
    status: str  # healthy, warning, critical
    recommendation: str


@dataclass
class VacuumReport:
    """Complete VACUUM/maintenance assessment report."""
    metrics: List[VacuumMetric]
    tables_needing_vacuum: List[Dict[str, Any]]
    tables_needing_analyze: List[Dict[str, Any]]
    autovacuum_status: Dict[str, Any]
    bloat_summary: Dict[str, Any]
    overall_health: str
    maintenance_priority: str
    timestamp: str


# -------------------------------------------------------------------
# VACUUM Signal Generators
# -------------------------------------------------------------------

class VacuumSignalGenerator:
    """
    Generate VACUUM/maintenance signals from PostgreSQL statistics.
    
    Detects:
    - High dead tuple ratios
    - Stale statistics
    - Table/index bloat
    - Autovacuum configuration issues
    - VACUUM progress stalls
    - Maintenance window needs
    """

    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

    def process(self, data: Any) -> SignalResult:
        """
        Process data and generate VACUUM signals.
        
        Args:
            data: PostgreSQL stats, table statistics, or JSON maintenance data
            
        Returns:
            SignalResult with VACUUM signals
        """
        start_time = time.time()
        signals: List[Signal] = []
        now = int(time.time() * 1000)

        # Handle different input types
        if isinstance(data, dict):
            # Check for different data structures
            if "table_statistics" in data or "tables" in data:
                signals = self._generate_from_table_stats(data)
            elif "pg_stat_user_tables" in data:
                signals = self._generate_from_pg_stats(data)
            elif "autovacuum" in data:
                signals = self._generate_from_autovacuum_config(data)
            else:
                signals = self._generate_from_generic_stats(data)
        elif isinstance(data, str):
            signals = self._generate_from_logs(data)
        else:
            self.logger.warning(f"Unknown data type for VACUUM analysis: {type(data)}")

        # Build analysis summary
        severity_count: Dict[str, int] = {}
        for s in signals:
            severity_count[s.severity] = severity_count.get(s.severity, 0) + 1

        analysis = {
            "signal_count": len(signals),
            "severities": severity_count,
            "highest_severity": max(severity_count, key=severity_count.get) if severity_count else "none",
            "vacuum_categories": self._count_categories(signals),
        }

        duration = time.time() - start_time
        self.logger.info(f"VACUUM analysis completed in {duration:.3f}s")

        return SignalResult(
            signals=signals,
            analysis=analysis,
            filtered_count=0,
            processing_time=duration,
        )

    def _generate_from_table_stats(self, data: Dict[str, Any]) -> List[Signal]:
        """Generate signals from table statistics."""
        signals: List[Signal] = []
        now = int(time.time() * 1000)

        tables = data.get("tables", data.get("table_statistics", {}).get("tables", []))
        diagnostic_hints = data.get("diagnostic_hints", [])

        for table in tables:
            table_name = table.get("table_name", "unknown")
            n_dead = table.get("n_dead_tup", 0)
            n_live = table.get("n_live_tup", 0)
            last_vacuum = table.get("last_vacuum", "")
            last_analyze = table.get("last_analyze", "")
            table_size = table.get("table_size_mb", 0)
            table_bloat = table.get("table_bloat_mb", 0)

            # Calculate dead tuple ratio
            total_tuples = n_dead + n_live
            dead_ratio = n_dead / total_tuples if total_tuples > 0 else 0

            # 1. High dead tuple ratio
            if dead_ratio > DEAD_TUPLE_THRESHOLD:
                signals.append(Signal(
                    id=f"high_dead_tuples_{table_name}_{now}",
                    name="high_dead_tuple_ratio",
                    type="vacuum",
                    severity="high" if dead_ratio > 0.4 else "medium",
                    confidence=0.92,
                    data={
                        "table_name": table_name,
                        "dead_tuples": n_dead,
                        "live_tuples": n_live,
                        "dead_tuple_ratio": round(dead_ratio, 3),
                        "threshold": DEAD_TUPLE_THRESHOLD,
                    },
                    metadata={
                        "explain": f"Table '{table_name}' has {dead_ratio:.1%} dead tuples (threshold: {DEAD_TUPLE_THRESHOLD:.0%})",
                        "category": "dead_tuple_management",
                        "recommendation": f"Run VACUUM (VERBOSE) {table_name};",
                        "remediation": "VACUUM will reclaim space and allow tuple reuse",
                    },
                ))

            # 2. Stale statistics (analyze)
            if last_analyze:
                analyze_age = self._parse_age(last_analyze)
                if analyze_age > STALE_STATS_THRESHOLD_DAYS:
                    signals.append(Signal(
                        id=f"stale_stats_{table_name}_{now}",
                        name="stale_table_statistics",
                        type="vacuum",
                        severity="medium",
                        confidence=0.90,
                        data={
                            "table_name": table_name,
                            "days_since_analyze": analyze_age,
                            "threshold": STALE_STATS_THRESHOLD_DAYS,
                            "last_analyze": last_analyze,
                        },
                        metadata={
                            "explain": f"ANALYZE not run on '{table_name}' for {analyze_age} days",
                            "category": "statistics_freshness",
                            "recommendation": f"Run ANALYZE {table_name};",
                            "remediation": "ANALYZE updates statistics for better query plans",
                        },
                    ))

            # 3. Table bloat
            if table_size > 0:
                bloat_ratio = table_bloat / table_size if table_bloat > 0 else 0
                if bloat_ratio > TABLE_BLOAT_THRESHOLD:
                    signals.append(Signal(
                        id=f"table_bloat_{table_name}_{now}",
                        name="high_table_bloat",
                        type="vacuum",
                        severity="high" if bloat_ratio > HIGH_BLOAT_THRESHOLD else "medium",
                        confidence=0.85,
                        data={
                            "table_name": table_name,
                            "table_size_mb": table_size,
                            "bloat_size_mb": table_bloat,
                            "bloat_ratio": round(bloat_ratio, 3),
                            "threshold": TABLE_BLOAT_THRESHOLD,
                        },
                        metadata={
                            "explain": f"Table '{table_name}' has {bloat_ratio:.1%} bloat ({table_bloat:.0f}MB)",
                            "category": "bloat_management",
                            "recommendation": "Consider VACUUM FULL or REINDEX during maintenance window",
                            "remediation": "VACUUM FULL reclaims space but requires exclusive lock",
                        },
                    ))

            # 4. Never vacuumed or long time since vacuum
            if last_vacuum:
                vacuum_age = self._parse_age(last_vacuum)
                if vacuum_age > 30:  # 30 days
                    signals.append(Signal(
                        id=f"old_vacuum_{table_name}_{now}",
                        name="vacuum_overdue",
                        type="vacuum",
                        severity="medium",
                        confidence=0.88,
                        data={
                            "table_name": table_name,
                            "days_since_vacuum": vacuum_age,
                            "last_vacuum": last_vacuum,
                        },
                        metadata={
                            "explain": f"Table '{table_name}' not vacuumed in {vacuum_age} days",
                            "category": "dead_tuple_management",
                            "recommendation": f"Run VACUUM (VERBOSE, ANALYZE) {table_name};",
                            "remediation": "Regular VACUUM prevents bloat accumulation",
                        },
                    ))
            elif n_dead > 10000:  # Never vacuumed but has dead tuples
                signals.append(Signal(
                    id=f"never_vacuumed_{table_name}_{now}",
                    name="vacuum_never_run",
                    type="vacuum",
                    severity="high",
                    confidence=0.90,
                    data={
                        "table_name": table_name,
                        "dead_tuples": n_dead,
                        "issue": "Table has never been vacuumed",
                    },
                    metadata={
                        "explain": f"Table '{table_name}' has {n_dead:,} dead tuples and no vacuum record",
                        "category": "dead_tuple_management",
                        "recommendation": f"Run VACUUM (VERBOSE) {table_name};",
                        "remediation": "Immediate VACUUM recommended to prevent performance degradation",
                    },
                ))

        # 5. Global diagnostic hints
        for hint in diagnostic_hints:
            hint_lower = hint.lower()
            if "high dead tuples" in hint_lower:
                signals.append(Signal(
                    id=f"global_high_dead_{now}",
                    name="high_dead_tuples_global",
                    type="vacuum",
                    severity="medium",
                    confidence=0.85,
                    data={
                        "diagnostic_hint": hint,
                    },
                    metadata={
                        "explain": f"Diagnostic indicates: {hint}",
                        "category": "dead_tuple_management",
                        "recommendation": "Review tables with highest dead tuple counts",
                    },
                ))

        return signals

    def _generate_from_pg_stats(self, data: Dict[str, Any]) -> List[Signal]:
        """Generate signals from pg_stat_user_tables data."""
        signals: List[Signal] = []
        now = int(time.time() * 1000)

        pg_stats = data.get("pg_stat_user_tables", [])
        autovacuum_settings = data.get("autovacuum_settings", {})

        for row in pg_stats:
            table_name = row.get("relname", "unknown")
            n_dead = row.get("n_dead_tup", 0)
            n_live = row.get("n_live_tup", 0)
            last_vacuum = row.get("last_vacuum") or row.get("last_autovacuum", "")
            last_analyze = row.get("last_analyze") or row.get("last_autoanalyze", "")
            
            # Calculate dead tuple ratio
            total = n_dead + n_live
            if total > 0:
                dead_ratio = n_dead / total
                
                if dead_ratio > DEAD_TUPLE_THRESHOLD:
                    signals.append(Signal(
                        id=f"pgstat_dead_{table_name}_{now}",
                        name="high_dead_tuple_ratio",
                        type="vacuum",
                        severity="high" if dead_ratio > 0.4 else "medium",
                        confidence=0.92,
                        data={
                            "table_name": table_name,
                            "dead_tuples": n_dead,
                            "dead_ratio": round(dead_ratio, 3),
                        },
                        metadata={
                            "explain": f"Table '{table_name}': {dead_ratio:.1%} dead tuples",
                            "category": "dead_tuple_management",
                        },
                    ))

        # 6. Autovacuum configuration check
        vacuum_scale = autovacuum_settings.get("autovacuum_vacuum_scale_factor", AUTOVACUUM_VACUUM_SCALE_FACTOR)
        analyze_scale = autovacuum_settings.get("autovacuum_analyze_scale_factor", AUTOVACUUM_ANALYZE_SCALE_FACTOR)
        naptime = autovacuum_settings.get("autovacuum_naptime", AUTOVACUUM_NAPTIME)
        workers = autovacuum_settings.get("autovacuum_max_workers", AUTOVACUUM_MAX_WORKERS)

        if vacuum_scale > 0.2:
            signals.append(Signal(
                id=f"autovacuum_scale_{now}",
                name="autovacuum_vacuum_scale_factor_high",
                type="vacuum",
                severity="medium",
                confidence=0.80,
                data={
                    "autovacuum_vacuum_scale_factor": vacuum_scale,
                    "default": AUTOVACUUM_VACUUM_SCALE_FACTOR,
                    "issue": "Autovacuum vacuum scale factor may be too conservative",
                },
                metadata={
                    "explain": f"autovacuum_vacuum_scale_factor={vacuum_scale} (default: {AUTOVACUUM_VACUUM_SCALE_FACTOR})",
                    "category": "autovacuum_configuration",
                    "recommendation": "Consider lowering to 0.01 for high-churn tables",
                },
            ))

        if workers < 3:
            signals.append(Signal(
                id=f"autovacuum_workers_{now}",
                name="insufficient_autovacuum_workers",
                type="vacuum",
                severity="low",
                confidence=0.75,
                data={
                    "autovacuum_max_workers": workers,
                    "recommended": 4,
                    "issue": "May not keep up with high write rates",
                },
                metadata={
                    "explain": f"autovacuum_max_workers={workers} may be insufficient",
                    "category": "autovacuum_configuration",
                    "recommendation": "Consider increasing to 4-6 for busy databases",
                },
            ))

        return signals

    def _generate_from_autovacuum_config(self, data: Dict[str, Any]) -> List[Signal]:
        """Generate signals from autovacuum configuration."""
        signals: List[Signal] = []
        now = int(time.time() * 1000)

        autovacuum = data.get("autovacuum", {})

        # Check if autovacuum is enabled
        enabled = autovacuum.get("autovacuum", "on")
        if str(enabled).lower() != "on":
            signals.append(Signal(
                id=f"autovacuum_disabled_{now}",
                name="autovacuum_disabled",
                type="vacuum",
                severity="critical",
                confidence=0.95,
                data={
                    "autovacuum_enabled": enabled,
                    "issue": "Autovacuum is disabled",
                },
                metadata={
                    "explain": "Autovacuum is disabled - manual VACUUM required",
                    "category": "autovacuum_configuration",
                    "recommendation": "Enable autovacuum: ALTER SYSTEM SET autovacuum = on;",
                    "remediation": "Autovacuum prevents bloat accumulation automatically",
                },
            ))

        # Check cost delay
        cost_delay = autovacuum.get("autovacuum_vacuum_cost_delay", "20ms")
        try:
            delay_ms = int(cost_delay.replace("ms", "").strip())
            if delay_ms > 20:
                signals.append(Signal(
                    id=f"autovacuum_cost_delay_{now}",
                    name="autovacuum_vacuum_cost_delay_high",
                    type="vacuum",
                    severity="low",
                    confidence=0.70,
                    data={
                        "autovacuum_vacuum_cost_delay": cost_delay,
                        "recommended": "2-10ms",
                    },
                    metadata={
                        "explain": f"autovacuum_vacuum_cost_delay={cost_delay} may slow VACUUM",
                        "category": "autovacuum_configuration",
                        "recommendation": "Lower to 2-10ms for busy databases",
                    },
                ))
        except ValueError:
            pass

        return signals

    def _generate_from_generic_stats(self, data: Dict[str, Any]) -> List[Signal]:
        """Generate signals from generic statistics data."""
        signals: List[Signal] = []
        now = int(time.time() * 1000)

        # Look for common patterns in the data
        for key, value in data.items():
            key_lower = key.lower()

            # Dead tuple ratio
            if "dead" in key_lower and "tuple" in key_lower:
                if isinstance(value, dict):
                    ratio = value.get("ratio", 0)
                    if ratio > DEAD_TUPLE_THRESHOLD:
                        signals.append(Signal(
                            id=f"generic_dead_{now}",
                            name="high_dead_tuple_ratio",
                            type="vacuum",
                            severity="high" if ratio > 0.4 else "medium",
                            confidence=0.88,
                            data={"metric": key, "value": value},
                            metadata={
                                "explain": f"Dead tuple ratio: {ratio:.1%}",
                                "category": "dead_tuple_management",
                            },
                        ))
                elif isinstance(value, (int, float)) and value > 100000:
                    signals.append(Signal(
                        id=f"generic_dead_count_{now}",
                        name="high_dead_tuple_count",
                        type="vacuum",
                        severity="medium",
                        confidence=0.85,
                        data={"metric": key, "value": value},
                        metadata={
                            "explain": f"High dead tuple count: {value:,}",
                            "category": "dead_tuple_management",
                        },
                    ))

            # Bloat
            if "bloat" in key_lower:
                if isinstance(value, dict):
                    bloat_ratio = value.get("ratio", 0) or value.get("percent", 0)
                    if bloat_ratio > TABLE_BLOAT_THRESHOLD:
                        signals.append(Signal(
                            id=f"generic_bloat_{now}",
                            name="high_table_bloat",
                            type="vacuum",
                            severity="high" if bloat_ratio > 0.5 else "medium",
                            confidence=0.85,
                            data={"metric": key, "value": value},
                            metadata={
                                "explain": f"Bloat ratio: {bloat_ratio:.1%}",
                                "category": "bloat_management",
                            },
                        ))

        return signals

    def _generate_from_logs(self, log_text: str) -> List[Signal]:
        """Generate signals from log text."""
        signals: List[Signal] = []
        now = int(time.time() * 1000)
        log_lower = log_text.lower()

        # VACUUM errors
        if "vacuum" in log_lower and "error" in log_lower:
            signals.append(Signal(
                id=f"vacuum_error_{now}",
                name="vacuum_error_detected",
                type="vacuum",
                severity="high",
                confidence=0.90,
                data={"pattern": "vacuum error"},
                metadata={
                    "explain": "VACUUM errors detected in logs",
                    "category": "vacuum_operations",
                    "remediation": "Check PostgreSQL logs for detailed error messages",
                },
            ))

        # Autovacuum messages
        if "autovacuum" in log_lower:
            if "skip" in log_lower or "skipping" in log_lower:
                signals.append(Signal(
                    id=f"autovacuum_skip_{now}",
                    name="autovacuum_skipping",
                    type="vacuum",
                    severity="medium",
                    confidence=0.85,
                    data={"pattern": "autovacuum skipping"},
                    metadata={
                        "explain": "Autovacuum is skipping tables - may indicate issues",
                        "category": "autovacuum_configuration",
                    },
                ))
            if "worker" in log_lower and ("started" in log_lower or "launched" in log_lower):
                signals.append(Signal(
                    id=f"autovacuum_worker_{now}",
                    name="autovacuum_worker_started",
                    type="vacuum",
                    severity="low",
                    confidence=0.70,
                    data={"pattern": "autovacuum worker"},
                    metadata={
                        "explain": "Autovacuum worker activity detected",
                        "category": "autovacuum_configuration",
                    },
                ))

        # Bloated index/table mentions
        if "bloat" in log_lower or "bloated" in log_lower:
            signals.append(Signal(
                id=f"bloat_mentioned_{now}",
                name="bloat_mentioned_in_logs",
                type="vacuum",
                severity="low",
                confidence=0.65,
                data={"pattern": "bloat mentioned"},
                metadata={
                    "explain": "Bloat mentioned in logs - may indicate maintenance needs",
                    "category": "bloat_management",
                },
            ))

        return signals

    def _count_categories(self, signals: List[Signal]) -> Dict[str, int]:
        """Count signals by VACUUM category."""
        categories: Dict[str, int] = {}
        for signal in signals:
            category = signal.metadata.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1
        return categories

    def _parse_age(self, timestamp_str: str) -> float:
        """Parse timestamp and return age in days."""
        if not timestamp_str:
            return float('inf')
        
        try:
            # Try ISO format
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            age = datetime.utcnow() - dt
            return age.days
        except ValueError:
            # Try other formats
            return 0.0

    def run_vacuum_check(
        self,
        table_stats: List[Dict[str, Any]],
        autovacuum_config: Optional[Dict[str, Any]] = None,
    ) -> VacuumReport:
        """
        Run comprehensive VACUUM check.
        
        Args:
            table_stats: List of table statistics from pg_stat_user_tables
            autovacuum_config: Autovacuum configuration settings
            
        Returns:
            VacuumReport with all metrics and recommendations
        """
        # Generate signals
        stats_data = {"tables": table_stats}
        if autovacuum_config:
            stats_data["autovacuum"] = autovacuum_config

        result = self.process(stats_data)

        # Convert signals to metrics
        metrics: List[VacuumMetric] = []
        tables_needing_vacuum: List[Dict[str, Any]] = []
        tables_needing_analyze: List[Dict[str, Any]] = []
        bloat_summary: Dict[str, Any] = {"tables": [], "total_bloat_mb": 0}
        autovacuum_status: Dict[str, Any] = {"enabled": True, "workers": 3}

        for table in table_stats:
            table_name = table.get("table_name", "unknown")
            n_dead = table.get("n_dead_tup", 0)
            n_live = table.get("n_live_tup", 0)
            total = n_dead + n_live
            dead_ratio = n_dead / total if total > 0 else 0
            last_analyze = table.get("last_analyze", "")
            table_bloat = table.get("table_bloat_mb", 0)
            table_size = table.get("table_size_mb", 0)

            # Add to tables needing vacuum
            if dead_ratio > DEAD_TUPLE_THRESHOLD:
                tables_needing_vacuum.append({
                    "table_name": table_name,
                    "dead_tuples": n_dead,
                    "dead_ratio": round(dead_ratio, 3),
                    "size_mb": table_size,
                })

            # Add to tables needing analyze
            if last_analyze:
                analyze_age = self._parse_age(last_analyze)
                if analyze_age > STALE_STATS_THRESHOLD_DAYS:
                    tables_needing_analyze.append({
                        "table_name": table_name,
                        "days_since_analyze": analyze_age,
                    })

            # Track bloat
            if table_bloat > 0:
                bloat_summary["tables"].append({
                    "table_name": table_name,
                    "bloat_mb": table_bloat,
                })
                bloat_summary["total_bloat_mb"] += table_bloat

        # Determine autovacuum status from config
        if autovacuum_config:
            autovacuum_status = {
                "enabled": autovacuum_config.get("autovacuum", "on") == "on",
                "workers": autovacuum_config.get("autovacuum_max_workers", 3),
                "vacuum_scale_factor": autovacuum_config.get("autovacuum_vacuum_scale_factor", 0.2),
                "analyze_scale_factor": autovacuum_config.get("autovacuum_analyze_scale_factor", 0.1),
                "cost_delay": autovacuum_config.get("autovacuum_vacuum_cost_delay", "20ms"),
            }

        # Determine overall health
        critical_count = sum(1 for s in result.signals if s.severity == "critical")
        high_count = sum(1 for s in result.signals if s.severity == "high")

        if critical_count > 0:
            overall_health = "critical"
            maintenance_priority = "immediate"
        elif high_count > 0 or len(tables_needing_vacuum) > 3:
            overall_health = "warning"
            maintenance_priority = "soon"
        elif len(tables_needing_vacuum) > 0:
            overall_health = "fair"
            maintenance_priority = "scheduled"
        else:
            overall_health = "healthy"
            maintenance_priority = "none"

        return VacuumReport(
            metrics=metrics,
            tables_needing_vacuum=tables_needing_vacuum,
            tables_needing_analyze=tables_needing_analyze,
            autovacuum_status=autovacuum_status,
            bloat_summary=bloat_summary,
            overall_health=overall_health,
            maintenance_priority=maintenance_priority,
            timestamp=datetime.utcnow().isoformat(),
        )


# -------------------------------------------------------------------
# Standalone Functions for Quick Checks
# -------------------------------------------------------------------

def check_dead_tuple_ratio(n_dead: int, n_live: int, threshold: float = DEAD_TUPLE_THRESHOLD) -> bool:
    """Check if dead tuple ratio exceeds threshold."""
    total = n_dead + n_live
    if total == 0:
        return False
    return (n_dead / total) > threshold


def check_stale_statistics(last_analyze: str, threshold_days: int = STALE_STATS_THRESHOLD_DAYS) -> bool:
    """Check if statistics are stale."""
    if not last_analyze:
        return True
    
    try:
        dt = datetime.fromisoformat(last_analyze.replace("Z", "+00:00"))
        age = datetime.utcnow() - dt
        return age.days > threshold_days
    except ValueError:
        return True


def calculate_vacuum_threshold(table_size: int, scale_factor: float = AUTOVACUUM_VACUUM_SCALE_FACTOR) -> int:
    """Calculate autovacuum threshold for a table."""
    # autovacuum_vacuum_threshold + scale_factor * table_size
    base_threshold = 50  # default autovacuum_vacuum_threshold
    return int(base_threshold + (scale_factor * table_size))


# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Example table statistics
    sample_tables = [
        {
            "table_name": "orders",
            "n_live_tup": 2250000,
            "n_dead_tup": 1840000,
            "dead_tuple_ratio": 0.45,
            "last_vacuum": "2026-01-05T02:12:00Z",
            "last_analyze": "2026-01-05T01:10:00Z",
            "table_size_mb": 2048,
            "table_bloat_mb": 1331,
        },
        {
            "table_name": "customers",
            "n_live_tup": 500000,
            "n_dead_tup": 12000,
            "dead_tuple_ratio": 0.02,
            "last_vacuum": "2026-01-26T02:05:00Z",
            "last_analyze": "2026-01-26T03:55:00Z",
            "table_size_mb": 256,
            "table_bloat_mb": 5,
        },
    ]

    autovacuum_config = {
        "autovacuum": "on",
        "autovacuum_max_workers": 3,
        "autovacuum_vacuum_scale_factor": 0.2,
        "autovacuum_vacuum_cost_delay": "20ms",
    }

    # Run VACUUM check
    generator = VacuumSignalGenerator()
    report = generator.run_vacuum_check(sample_tables, autovacuum_config)

    print(f"Overall Health: {report.overall_health}")
    print(f"Maintenance Priority: {report.maintenance_priority}")
    print(f"Tables needing VACUUM: {len(report.tables_needing_vacuum)}")
    print(f"Tables needing ANALYZE: {len(report.tables_needing_analyze)}")
    print(f"Total bloat: {report.bloat_summary['total_bloat_mb']:.0f} MB")

