#!/usr/bin/env python3
"""
VACUUM Recommender Module

Generate DBA-safe VACUUM/maintenance recommendations for PostgreSQL.
Designed for optimal database maintenance with proper safety measures.

Features:
- VACUUM timing recommendations
- Autovacuum tuning
- Bloat management
- Maintenance window planning
- Online vs offline operations
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from signals.vacuum_signals import VacuumReport, VacuumMetric
from signals.signal_engine import SignalResult, Signal


# -------------------------------------------------------------------
# Data Models
# -------------------------------------------------------------------

@dataclass
class VacuumAction:
    """
    A single VACUUM action with full safety information.
    """
    action: str
    sql: Optional[str]
    risk_level: str  # low, medium, high
    is_online: bool  # Can run while database is active
    requires_approval: bool
    estimated_downtime: str
    rollback_notes: str
    priority: str  # immediate, soon, scheduled
    table_name: Optional[str]
    verification_command: Optional[str]


@dataclass
class VacuumRecommendation:
    """
    A complete VACUUM recommendation with actions.
    """
    check_name: str
    category: str
    severity: str
    title: str
    description: str
    current_state: str
    desired_state: str
    actions: List[VacuumAction]
    confidence: float
    table_name: Optional[str]
    estimated_improvement: str
    references: List[str]


@dataclass
class VacuumRecommendationReport:
    """
    Complete VACUUM recommendation report.
    """
    recommendations: List[VacuumRecommendation]
    overall_health: str
    health_score: float
    tables_needing_vacuum: List[Dict[str, Any]]
    tables_needing_analyze: List[Dict[str, Any]]
    autovacuum_config_issues: List[Dict[str, Any]]
    bloat_summary: Dict[str, Any]
    maintenance_plan: List[Dict[str, Any]]
    total_dead_tuples: int
    total_bloat_mb: float
    timestamp: str


# -------------------------------------------------------------------
# VACUUM Recommender
# -------------------------------------------------------------------

class VacuumRecommender:
    """
    Generate DBA-safe VACUUM recommendations.
    
    Maps VACUUM signals to actionable recommendations with:
    - Proper risk assessment
    - Online/offline operation guidance
    - Autovacuum tuning
    - Maintenance window planning
    """

    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

    def recommend(
        self,
        signal_result: SignalResult,
        vacuum_report: Optional[VacuumReport] = None,
    ) -> VacuumRecommendationReport:
        """
        Generate VACUUM recommendations.
        
        Args:
            signal_result: SignalResult from VacuumSignalGenerator
            vacuum_report: Optional VacuumReport for detailed analysis
            
        Returns:
            VacuumRecommendationReport with all recommendations
        """
        recommendations: List[VacuumRecommendation] = []

        # Process each signal
        for signal in signal_result.signals:
            rec = self._signal_to_recommendation(signal)
            if rec:
                recommendations.append(rec)

        # Calculate counts from vacuum_report if available
        tables_needing_vacuum = vacuum_report.tables_needing_vacuum if vacuum_report else []
        tables_needing_analyze = vacuum_report.tables_needing_analyze if vacuum_report else []
        bloat_summary = vacuum_report.bloat_summary if vacuum_report else {"total_bloat_mb": 0, "tables": []}
        autovacuum_issues = self._extract_autovacuum_issues(signal_result)

        # Calculate health score
        total_checks = len(recommendations)
        if total_checks > 0:
            score = sum(
                4 - {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(r.severity, 0)
                for r in recommendations
            )
            max_score = total_checks * 4
            health_score = (score / max_score) * 100
        else:
            health_score = 100.0

        # Determine overall health
        critical_count = sum(1 for r in recommendations if r.severity == "critical")
        high_count = sum(1 for r in recommendations if r.severity == "high")

        if critical_count > 0:
            overall_health = "critical"
        elif high_count > 0 or len(tables_needing_vacuum) > 3:
            overall_health = "warning"
        elif len(tables_needing_vacuum) > 0:
            overall_health = "fair"
        else:
            overall_health = "healthy"

        # Build maintenance plan
        maintenance_plan = self._build_maintenance_plan(
            recommendations,
            tables_needing_vacuum,
            tables_needing_analyze,
            autovacuum_issues,
        )

        # Calculate totals
        total_dead = sum(t.get("dead_tuples", 0) for t in tables_needing_vacuum)
        total_bloat = bloat_summary.get("total_bloat_mb", 0)

        return VacuumRecommendationReport(
            recommendations=recommendations,
            overall_health=overall_health,
            health_score=health_score,
            tables_needing_vacuum=tables_needing_vacuum,
            tables_needing_analyze=tables_needing_analyze,
            autovacuum_config_issues=autovacuum_issues,
            bloat_summary=bloat_summary,
            maintenance_plan=maintenance_plan,
            total_dead_tuples=total_dead,
            total_bloat_mb=total_bloat,
            timestamp=datetime.utcnow().isoformat(),
        )

    def _signal_to_recommendation(self, signal: Signal) -> Optional[VacuumRecommendation]:
        """Convert a VACUUM signal to a recommendation."""
        signal_map = {
            "high_dead_tuple_ratio": self._recommend_vacuum,
            "high_dead_tuples_global": self._recommend_vacuum,
            "stale_table_statistics": self._recommend_analyze,
            "high_table_bloat": self._recommend_bloat_removal,
            "vacuum_overdue": self._recommend_scheduled_vacuum,
            "vacuum_never_run": self._recommend_urgent_vacuum,
            "autovacuum_disabled": self._recommend_autovacuum_enable,
            "autovacuum_vacuum_scale_factor_high": self._recommend_autovacuum_tuning,
            "autovacuum_vacuum_cost_delay_high": self._recommend_autovacuum_cost_delay,
            "insufficient_autovacuum_workers": self._recommend_autovacuum_workers,
            "vacuum_error_detected": self._recommend_vacuum_error,
            "autovacuum_skipping": self._recommend_autovacuum_skip,
            "bloat_mentioned_in_logs": self._recommend_bloat_check,
        }

        handler = signal_map.get(signal.name)
        if handler:
            return handler(signal)

        return None

    def _recommend_vacuum(self, signal: Signal) -> VacuumRecommendation:
        """Recommendation for VACUUM operation."""
        table_name = signal.data.get("table_name", "unknown")
        dead_ratio = signal.data.get("dead_tuple_ratio", 0)
        dead_tuples = signal.data.get("dead_tuples", 0)

        return VacuumRecommendation(
            check_name=signal.name,
            category="dead_tuple_management",
            severity=signal.severity,
            title=f"Run VACUUM on Table '{table_name}'",
            description=f"Table has {dead_ratio:.1%} dead tuples - VACUUM needed to reclaim space.",
            current_state=f"Dead tuple ratio: {dead_ratio:.1%} ({dead_tuples:,} dead tuples)",
            desired_state="Dead tuple ratio below 20%",
            actions=[
                VacuumAction(
                    action="Run VACUUM VERBOSE ANALYZE",
                    sql=f"VACUUM (VERBOSE, ANALYZE) {table_name};",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none (short locks only)",
                    rollback_notes="Not needed - VACUUM is a recovery operation",
                    priority="immediate",
                    table_name=table_name,
                    verification_command=f"SELECT n_dead_tup, n_live_tup FROM pg_stat_user_tables WHERE relname = '{table_name}';",
                ),
                VacuumAction(
                    action="Monitor VACUUM progress",
                    sql="SELECT * FROM pg_stat_progress_vacuum;",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not applicable - read operation",
                    priority="immediate",
                    table_name=table_name,
                    verification_command="SELECT * FROM pg_stat_progress_vacuum WHERE relname = '{table_name}';",
                ),
            ],
            confidence=signal.confidence,
            table_name=table_name,
            estimated_improvement=f"Reduce dead tuples by {dead_tuples:,} and improve query plans",
            references=[
                "https://www.postgresql.org/docs/current/sql-vacuum.html",
            ],
        )

    def _recommend_analyze(self, signal: Signal) -> VacuumRecommendation:
        """Recommendation for ANALYZE operation."""
        table_name = signal.data.get("table_name", "unknown")
        days = signal.data.get("days_since_analyze", 0)

        return VacuumRecommendation(
            check_name=signal.name,
            category="statistics_freshness",
            severity=signal.severity,
            title=f"Run ANALYZE on Table '{table_name}'",
            description=f"Statistics are {days} days old - ANALYZE needed for query optimization.",
            current_state=f"Last ANALYZE: {days} days ago",
            desired_state="Statistics less than 7 days old",
            actions=[
                VacuumAction(
                    action="Run ANALYZE",
                    sql=f"ANALYZE {table_name};",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not needed - ANALYZE only collects statistics",
                    priority="immediate",
                    table_name=table_name,
                    verification_command=f"SELECT last_analyze FROM pg_stat_user_tables WHERE relname = '{table_name}';",
                ),
                VacuumAction(
                    action="Consider auto-analyze on vacuum",
                    sql=f"VACUUM (VERBOSE, ANALYZE) {table_name};",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not applicable",
                    priority="soon",
                    table_name=table_name,
                    verification_command=f"SELECT last_analyze, last_vacuum FROM pg_stat_user_tables WHERE relname = '{table_name}';",
                ),
            ],
            confidence=signal.confidence,
            table_name=table_name,
            estimated_improvement="Improved query plan accuracy and execution time",
            references=[
                "https://www.postgresql.org/docs/current/sql-analyze.html",
            ],
        )

    def _recommend_bloat_removal(self, signal: Signal) -> VacuumRecommendation:
        """Recommendation for bloat removal."""
        table_name = signal.data.get("table_name", "unknown")
        bloat_size = signal.data.get("bloat_size_mb", 0)
        bloat_ratio = signal.data.get("bloat_ratio", 0)

        return VacuumRecommendation(
            check_name=signal.name,
            category="bloat_management",
            severity=signal.severity,
            title=f"Address Table Bloat on '{table_name}'",
            description=f"Table has {bloat_ratio:.1%} bloat ({bloat_size:.0f}MB) - space reclamation needed.",
            current_state=f"Bloat: {bloat_ratio:.1%} ({bloat_size:.0f}MB)",
            desired_state="Bloat below 30%",
            actions=[
                VacuumAction(
                    action="Run VACUUM to reclaim space",
                    sql=f"VACUUM (VERBOSE) {table_name};",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none (short locks)",
                    rollback_notes="Not applicable - VACUUM is recovery operation",
                    priority="soon",
                    table_name=table_name,
                    verification_command=f"SELECT pg_size_pretty(pg_relation_size('{table_name}'));",
                ),
                VacuumAction(
                    action="Consider VACUUM FULL during maintenance window",
                    sql=f"VACUUM FULL {table_name};",
                    risk_level="high",
                    is_online=False,
                    requires_approval=True,
                    estimated_downtime="requires exclusive lock",
                    rollback_notes="Not applicable - space already reclaimed",
                    priority="scheduled",
                    table_name=table_name,
                    verification_command=f"SELECT pg_size_pretty(pg_relation_size('{table_name}')) BEFORE vs AFTER;",
                ),
                VacuumAction(
                    action="Consider REINDEX to remove index bloat",
                    sql=f"REINDEX TABLE {table_name};",
                    risk_level="medium",
                    is_online=False,
                    requires_approval=True,
                    estimated_downtime="brief exclusive lock",
                    rollback_notes="Not applicable - REINDEX is recovery",
                    priority="scheduled",
                    table_name=table_name,
                    verification_command=f"SELECT pg_size_pretty(pg_relation_size(indexrelid)) FROM pg_stat_user_indexes WHERE relname = '{table_name}';",
                ),
            ],
            confidence=signal.confidence,
            table_name=table_name,
            estimated_improvement=f"Reclaim {bloat_size:.0f}MB of disk space",
            references=[
                "https://www.postgresql.org/docs/current/routine-vacuuming.html",
                "https://www.postgresql.org/docs/current/sql-reindex.html",
            ],
        )

    def _recommend_scheduled_vacuum(self, signal: Signal) -> VacuumRecommendation:
        """Recommendation for overdue vacuum."""
        table_name = signal.data.get("table_name", "unknown")
        days = signal.data.get("days_since_vacuum", 0)

        return VacuumRecommendation(
            check_name=signal.name,
            category="dead_tuple_management",
            severity=signal.severity,
            title=f"Schedule VACUUM for Table '{table_name}'",
            description=f"Table not vacuumed in {days} days - schedule maintenance.",
            current_state=f"Last VACUUM: {days} days ago",
            desired_state="Regular VACUUM schedule maintained",
            actions=[
                VacuumAction(
                    action="Schedule manual VACUUM",
                    sql=f"VACUUM (VERBOSE, ANALYZE) {table_name};",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not applicable",
                    priority="soon",
                    table_name=table_name,
                    verification_command=f"SELECT last_vacuum FROM pg_stat_user_tables WHERE relname = '{table_name}';",
                ),
                VacuumAction(
                    action="Review autovacuum settings for this table",
                    sql=f"SELECT * FROM pg_stat_user_tables WHERE relname = '{table_name}';",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not applicable",
                    priority="scheduled",
                    table_name=table_name,
                    verification_command="SHOW autovacuum_vacuum_scale_factor;",
                ),
            ],
            confidence=signal.confidence,
            table_name=table_name,
            estimated_improvement="Prevent bloat accumulation and maintain performance",
            references=[
                "https://www.postgresql.org/docs/current/runtime-config-autovacuum.html",
            ],
        )

    def _recommend_urgent_vacuum(self, signal: Signal) -> VacuumRecommendation:
        """Recommendation for never-vacuumed tables."""
        table_name = signal.data.get("table_name", "unknown")
        dead_tuples = signal.data.get("dead_tuples", 0)

        return VacuumRecommendation(
            check_name=signal.name,
            category="dead_tuple_management",
            severity=signal.severity,
            title=f"Urgent: VACUUM Never Run on '{table_name}'",
            description=f"Table has {dead_tuples:,} dead tuples and never been vacuumed.",
            current_state="No VACUUM record found",
            desired_state="Regular VACUUM maintenance",
            actions=[
                VacuumAction(
                    action="Run immediate VACUUM",
                    sql=f"VACUUM (VERBOSE, ANALYZE) {table_name};",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not applicable",
                    priority="immediate",
                    table_name=table_name,
                    verification_command=f"SELECT n_dead_tup FROM pg_stat_user_tables WHERE relname = '{table_name}';",
                ),
                VacuumAction(
                    action="Enable autovacuum for table",
                    sql=f"ALTER TABLE {table_name} SET (autovacuum_enabled = true);",
                    risk_level="low",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none",
                    rollback_notes=f"ALTER TABLE {table_name} SET (autovacuum_enabled = false);",
                    priority="immediate",
                    table_name=table_name,
                    verification_command=f"SELECT reloptions FROM pg_class WHERE relname = '{table_name}';",
                ),
            ],
            confidence=signal.confidence,
            table_name=table_name,
            estimated_improvement=f"Remove {dead_tuples:,} dead tuples and improve performance",
            references=[
                "https://www.postgresql.org/docs/current/sql-vacuum.html",
            ],
        )

    def _recommend_autovacuum_enable(self, signal: Signal) -> VacuumRecommendation:
        """Recommendation for enabling autovacuum."""
        return VacuumRecommendation(
            check_name=signal.name,
            category="autovacuum_configuration",
            severity=signal.severity,
            title="Enable Autovacuum",
            description="Autovacuum is currently disabled - manual VACUUM required.",
            current_state="autovacuum = off",
            desired_state="autovacuum = on with appropriate settings",
            actions=[
                VacuumAction(
                    action="Enable autovacuum",
                    sql="ALTER SYSTEM SET autovacuum = on; SELECT pg_reload_conf();",
                    risk_level="medium",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none (requires reload)",
                    rollback_notes="ALTER SYSTEM SET autovacuum = off;",
                    priority="immediate",
                    table_name=None,
                    verification_command="SHOW autovacuum;",
                ),
                VacuumAction(
                    action="Configure autovacuum settings",
                    sql="ALTER SYSTEM SET autovacuum_vacuum_threshold = 50; ALTER SYSTEM SET autovacuum_vacuum_scale_factor = 0.01; SELECT pg_reload_conf();",
                    risk_level="medium",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none",
                    rollback_notes="ALTER SYSTEM RESET autovacuum_vacuum_threshold; ALTER SYSTEM RESET autovacuum_vacuum_scale_factor;",
                    priority="soon",
                    table_name=None,
                    verification_command="SHOW autovacuum_vacuum_scale_factor;",
                ),
            ],
            confidence=signal.confidence,
            table_name=None,
            estimated_improvement="Automatic maintenance and bloat prevention",
            references=[
                "https://www.postgresql.org/docs/current/runtime-config-autovacuum.html",
            ],
        )

    def _recommend_autovacuum_tuning(self, signal: Signal) -> VacuumRecommendation:
        """Recommendation for autovacuum scale factor tuning."""
        current_value = signal.data.get("autovacuum_vacuum_scale_factor", 0.2)

        return VacuumRecommendation(
            check_name=signal.name,
            category="autovacuum_configuration",
            severity=signal.severity,
            title="Tune Autovacuum Scale Factor",
            description=f"autovacuum_vacuum_scale_factor = {current_value} may be too conservative.",
            current_state=f"autovacuum_vacuum_scale_factor = {current_value}",
            desired_state="autovacuum_vacuum_scale_factor = 0.01 (or lower for high-churn tables)",
            actions=[
                VacuumAction(
                    action="Reduce scale factor for more frequent VACUUM",
                    sql="ALTER SYSTEM SET autovacuum_vacuum_scale_factor = 0.01; SELECT pg_reload_conf();",
                    risk_level="medium",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none",
                    rollback_notes="ALTER SYSTEM SET autovacuum_vacuum_scale_factor = 0.2;",
                    priority="soon",
                    table_name=None,
                    verification_command="SHOW autovacuum_vacuum_scale_factor;",
                ),
                VacuumAction(
                    action="Set per-table autovacuum settings for high-churn tables",
                    sql=f"ALTER TABLE high_churn_table SET (autovacuum_vacuum_scale_factor = 0.005);",
                    risk_level="low",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none",
                    rollback_notes=f"ALTER TABLE high_churn_table RESET (autovacuum_vacuum_scale_factor);",
                    priority="scheduled",
                    table_name=None,
                    verification_command="SELECT relname, reloptions FROM pg_class WHERE reloptions IS NOT NULL;",
                ),
            ],
            confidence=signal.confidence,
            table_name=None,
            estimated_improvement="More responsive autovacuum for high-write tables",
            references=[
                "https://www.postgresql.org/docs/current/runtime-config-autovacuum.html",
            ],
        )

    def _recommend_autovacuum_cost_delay(self, signal: Signal) -> VacuumRecommendation:
        """Recommendation for autovacuum cost delay."""
        current = signal.data.get("autovacuum_vacuum_cost_delay", "20ms")

        return VacuumRecommendation(
            check_name=signal.name,
            category="autovacuum_configuration",
            severity=signal.severity,
            title="Optimize Autovacuum Cost Delay",
            description=f"autovacuum_vacuum_cost_delay = {current} may slow VACUUM too much.",
            current_state=f"autovacuum_vacuum_cost_delay = {current}",
            desired_state="autovacuum_vacuum_cost_delay = 2-10ms for responsive VACUUM",
            actions=[
                VacuumAction(
                    action="Reduce cost delay for faster VACUUM",
                    sql="ALTER SYSTEM SET autovacuum_vacuum_cost_delay = '2ms'; SELECT pg_reload_conf();",
                    risk_level="medium",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none",
                    rollback_notes="ALTER SYSTEM SET autovacuum_vacuum_cost_delay = '20ms';",
                    priority="scheduled",
                    table_name=None,
                    verification_command="SHOW autovacuum_vacuum_cost_delay;",
                ),
                VacuumAction(
                    action="Set cost limit for more aggressive VACUUM",
                    sql="ALTER SYSTEM SET autovacuum_vacuum_cost_limit = 1000; SELECT pg_reload_conf();",
                    risk_level="medium",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none",
                    rollback_notes="ALTER SYSTEM RESET autovacuum_vacuum_cost_limit;",
                    priority="scheduled",
                    table_name=None,
                    verification_command="SHOW autovacuum_vacuum_cost_limit;",
                ),
            ],
            confidence=signal.confidence,
            table_name=None,
            estimated_improvement="Faster autovacuum completion with minimal performance impact",
            references=[
                "https://www.postgresql.org/docs/current/runtime-config-autovacuum.html",
            ],
        )

    def _recommend_autovacuum_workers(self, signal: Signal) -> VacuumRecommendation:
        """Recommendation for autovacuum workers."""
        current = signal.data.get("autovacuum_max_workers", 3)

        return VacuumRecommendation(
            check_name=signal.name,
            category="autovacuum_configuration",
            severity=signal.severity,
            title="Increase Autovacuum Workers",
            description=f"autovacuum_max_workers = {current} may not keep up with write rate.",
            current_state=f"autovacuum_max_workers = {current}",
            desired_state="4-6 workers for busy databases",
            actions=[
                VacuumAction(
                    action="Increase autovacuum workers",
                    sql="ALTER SYSTEM SET autovacuum_max_workers = 4; SELECT pg_reload_conf();",
                    risk_level="high",
                    is_online=False,
                    requires_approval=True,
                    estimated_downtime="requires restart",
                    rollback_notes="ALTER SYSTEM SET autovacuum_max_workers = 3;",
                    priority="scheduled",
                    table_name=None,
                    verification_command="SHOW autovacuum_max_workers;",
                ),
            ],
            confidence=signal.confidence,
            table_name=None,
            estimated_improvement="Better parallel VACUUM processing",
            references=[
                "https://www.postgresql.org/docs/current/runtime-config-autovacuum.html",
            ],
        )

    def _recommend_vacuum_error(self, signal: Signal) -> VacuumRecommendation:
        """Recommendation for vacuum errors."""
        return VacuumRecommendation(
            check_name=signal.name,
            category="vacuum_operations",
            severity=signal.severity,
            title="Investigate and Resolve VACUUM Errors",
            description="VACUUM errors detected - investigate and resolve before next maintenance.",
            current_state="VACUUM errors in logs",
            desired_state="Healthy VACUUM operations",
            actions=[
                VacuumAction(
                    action="Review detailed error logs",
                    sql="SELECT * FROM pg_log WHERE message ILIKE '%vacuum%' AND message ILIKE '%error%' ORDER BY log_time DESC LIMIT 20;",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not applicable",
                    priority="immediate",
                    table_name=None,
                    verification_command="SELECT * FROM pg_log WHERE message ILIKE '%vacuum%' ORDER BY log_time DESC LIMIT 10;",
                ),
                VacuumAction(
                    action="Check for corrupted data",
                    sql="SELECT * FROM pg_stat_all_tables WHERE n_tup_del > 0;",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not applicable",
                    priority="immediate",
                    table_name=None,
                    verification_command="SELECT relname, n_tup_ins, n_tup_upd, n_tup_del FROM pg_stat_user_tables;",
                ),
                VacuumAction(
                    action="Try VACUUM on individual problem table",
                    sql="VACUUM (VERBOSE, SKIP_LOCKED) problem_table;",
                    risk_level="low",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none",
                    rollback_notes="Not applicable",
                    priority="soon",
                    table_name=None,
                    verification_command="SELECT * FROM pg_stat_progress_vacuum;",
                ),
            ],
            confidence=signal.confidence,
            table_name=None,
            estimated_improvement="Resolved VACUUM issues and healthy maintenance",
            references=[
                "https://www.postgresql.org/docs/current/sql-vacuum.html",
            ],
        )

    def _recommend_autovacuum_skip(self, signal: Signal) -> VacuumRecommendation:
        """Recommendation for autovacuum skipping."""
        return VacuumRecommendation(
            check_name=signal.name,
            category="autovacuum_configuration",
            severity=signal.severity,
            title="Investigate Autovacuum Skipping Tables",
            description="Autovacuum is skipping some tables - may indicate configuration issues.",
            current_state="Autovacuum skipping tables",
            desired_state="Autovacuum processing all tables as needed",
            actions=[
                VacuumAction(
                    action="Check table-specific autovacuum settings",
                    sql="SELECT relname, reloptions, pg_stat_user_tables.* FROM pg_stat_user_tables LEFT JOIN pg_class ON pg_stat_user_tables.relid = pg_class.oid WHERE reloptions IS NOT NULL;",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not applicable",
                    priority="soon",
                    table_name=None,
                    verification_command="SELECT relname, reloptions FROM pg_class WHERE reloptions IS NOT NULL;",
                ),
                VacuumAction(
                    action="Review table freeze age settings",
                    sql="SHOW vacuum_freeze_table_age; SHOW vacuum_freeze_min_age;",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not applicable",
                    priority="soon",
                    table_name=None,
                    verification_command="SHOW vacuum_freeze_table_age;",
                ),
            ],
            confidence=signal.confidence,
            table_name=None,
            estimated_improvement="Comprehensive autovacuum coverage",
            references=[
                "https://www.postgresql.org/docs/current/runtime-config-autovacuum.html",
            ],
        )

    def _recommend_bloat_check(self, signal: Signal) -> VacuumRecommendation:
        """Recommendation for bloat check."""
        return VacuumRecommendation(
            check_name=signal.name,
            category="bloat_management",
            severity=signal.severity,
            title="Perform Comprehensive Bloat Check",
            description="Bloat mentioned in logs - schedule thorough investigation.",
            current_state="Potential bloat detected",
            desired_state="Verified healthy table/index bloat levels",
            actions=[
                VacuumAction(
                    action="Check table and index bloat",
                    sql="SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size, pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size FROM pg_stat_user_tables ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC LIMIT 20;",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not applicable",
                    priority="soon",
                    table_name=None,
                    verification_command="SELECT * FROM pg_stat_user_tables ORDER BY n_dead_tup DESC LIMIT 10;",
                ),
                VacuumAction(
                    action="Use pgstattuple extension for accurate bloat",
                    sql="SELECT * FROM pgstattuple('table_name');",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not applicable",
                    priority="scheduled",
                    table_name=None,
                    verification_command="SELECT * FROM pgstattuple('pg_proc');",
                ),
            ],
            confidence=signal.confidence,
            table_name=None,
            estimated_improvement="Accurate bloat assessment and targeted cleanup",
            references=[
                "https://www.postgresql.org/docs/current/pgstattuple.html",
            ],
        )

    def _extract_autovacuum_issues(self, signal_result: SignalResult) -> List[Dict[str, Any]]:
        """Extract autovacuum configuration issues from signals."""
        issues = []
        for signal in signal_result.signals:
            if "autovacuum" in signal.name.lower():
                issues.append({
                    "signal": signal.name,
                    "severity": signal.severity,
                    "data": signal.data,
                })
        return issues

    def _build_maintenance_plan(
        self,
        recommendations: List[VacuumRecommendation],
        tables_needing_vacuum: List[Dict[str, Any]],
        tables_needing_analyze: List[Dict[str, Any]],
        autovacuum_issues: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build phased maintenance plan."""
        plan = []

        # Phase 1: Immediate (Critical issues)
        phase1 = [r for r in recommendations if r.severity == "critical"]
        if phase1:
            plan.append({
                "phase": 1,
                "name": "Critical VACUUM Issues",
                "priority": "immediate",
                "description": "Address critical issues immediately",
                "actions": [
                    {
                        "type": a.action,
                        "sql": a.sql,
                        "table": a.table_name,
                    }
                    for r in phase1 for a in r.actions
                ],
            })

        # Phase 2: Soon (High priority)
        phase2 = [r for r in recommendations if r.severity == "high"]
        if phase2:
            plan.append({
                "phase": 2,
                "name": "High Priority Maintenance",
                "priority": "soon",
                "description": "Address high priority items within 24 hours",
                "tables_vacuum": [t["table_name"] for t in tables_needing_vacuum[:5]],
                "tables_analyze": [t["table_name"] for t in tables_needing_analyze[:5]],
            })

        # Phase 3: Scheduled (Medium/Low)
        phase3 = [r for r in recommendations if r.severity in ["medium", "low"]]
        if phase3:
            plan.append({
                "phase": 3,
                "name": "Regular Maintenance",
                "priority": "scheduled",
                "description": "Schedule during next maintenance window",
                "actions": [
                    {
                        "type": a.action,
                        "priority": a.priority,
                    }
                    for r in phase3 for a in r.actions
                ],
            })

        # Phase 4: Autovacuum tuning
        if autovacuum_issues:
            plan.append({
                "phase": 4,
                "name": "Autovacuum Optimization",
                "priority": "scheduled",
                "description": "Tune autovacuum configuration",
                "issues": autovacuum_issues,
            })

        return plan


# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------

if __name__ == "__main__":
    from signals.vacuum_signals import VacuumSignalGenerator

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
    ]

    # Generate vacuum report
    generator = VacuumSignalGenerator()
    signal_result = generator.process({"tables": sample_tables})

    # Get recommendations
    recommender = VacuumRecommender()
    report = recommender.recommend(signal_result)

    print(f"Health Score: {report.health_score:.1f}%")
    print(f"Overall Health: {report.overall_health}")
    print(f"Tables needing VACUUM: {len(report.tables_needing_vacuum)}")
    print(f"Tables needing ANALYZE: {len(report.tables_needing_analyze)}")
    print(f"Total Dead Tuples: {report.total_dead_tuples:,}")
    print(f"Total Bloat: {report.total_bloat_mb:.0f} MB")

