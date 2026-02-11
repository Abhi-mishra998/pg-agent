#!/usr/bin/env python3
"""
SignalEngine - Signal Processing Pipeline (v4)

Generate → Analyze → Filter

Enhanced for:
- SQL destructive operations
- PostgreSQL incidents
- pgbench performance analysis
- Explainable, deterministic signals
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Callable

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

SEVERITY_LEVELS = {"low", "medium", "high", "critical"}

# pgbench thresholds (industry-aligned defaults)
PG_LOW_TPS = 100
PG_HIGH_LATENCY_MS = 200

# -------------------------------------------------------------------
# Data Models
# -------------------------------------------------------------------

@dataclass
class Signal:
    id: str
    name: str
    type: str
    confidence: float
    severity: str
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.severity not in SEVERITY_LEVELS:
            raise ValueError(
                f"Invalid severity '{self.severity}'. "
                f"Must be one of {SEVERITY_LEVELS}"
            )


@dataclass
class SignalResult:
    signals: List[Signal]
    analysis: Dict[str, Any]
    filtered_count: int
    processing_time: float


# -------------------------------------------------------------------
# Signal Engine
# -------------------------------------------------------------------

class SignalEngine:
    """
    Deterministic signal processing engine.

    Responsibilities:
    - Detect incidents
    - Produce explainable signals
    - NEVER hallucinate
    """

    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        self._generators: List[Callable] = []
        self._analyzers: List[Callable] = []
        self._filters: List[Callable] = []

        self._register_default_handlers()

    # -------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------

    def _register_default_handlers(self) -> None:
        self._generators.extend([
            self._generate_text_signals,
            self._generate_sql_destructive_signals,
            self._generate_pgbench_signals,
            self._generate_json_context_signals,  # Analyze JSON context (simple format)
            self._generate_incident_signals,  # Analyze incident/alert JSON format
            self._generate_index_health_signals,  # Analyze index health JSON format
            self._generate_query_metrics_signals,  # Analyze query metrics
            self._generate_table_stats_signals,  # Analyze table statistics
        ])
        self._analyzers.append(self._analyze_metadata)
        self._filters.append(self._filter_low_confidence)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def process(self, data: Any) -> SignalResult:
        start = time.time()
        self.logger.info("Starting signal processing pipeline")

        signals = self._generate(data)
        self._analyze(signals, data)

        filtered, removed = self._filter(signals)
        analysis = self._build_analysis(filtered)

        duration = time.time() - start
        self.logger.info("Signal processing completed in %.3fs", duration)

        return SignalResult(
            signals=filtered,
            analysis=analysis,
            filtered_count=removed,
            processing_time=duration,
        )

    # -------------------------------------------------------------------
    # Pipeline Stages
    # -------------------------------------------------------------------

    def _generate(self, data: Any) -> List[Signal]:
        signals: List[Signal] = []
        for gen in self._generators:
            try:
                signals.extend(gen(data))
            except Exception as e:
                self.logger.error("Generator error: %s", e)
        self.logger.info("Generated %d signals", len(signals))
        return signals

    def _analyze(self, signals: List[Signal], data: Any) -> None:
        for sig in signals:
            for analyzer in self._analyzers:
                analyzer(sig, data)

    def _filter(self, signals: List[Signal]) -> tuple[List[Signal], int]:
        original = len(signals)
        filtered = signals
        for f in self._filters:
            filtered = f(filtered)
        return filtered, original - len(filtered)

    # -------------------------------------------------------------------
    # Analysis Summary
    # -------------------------------------------------------------------

    def _build_analysis(self, signals: List[Signal]) -> Dict[str, Any]:
        if not signals:
            return {
                "signal_count": 0,
                "summary": "No incidents detected",
            }

        severity_count: Dict[str, int] = {}
        for s in signals:
            severity_count[s.severity] = severity_count.get(s.severity, 0) + 1

        return {
            "signal_count": len(signals),
            "severities": severity_count,
            "highest_severity": max(severity_count, key=severity_count.get),
        }

    # -------------------------------------------------------------------
    # Generators
    # -------------------------------------------------------------------

    def _generate_text_signals(self, data: Any) -> List[Signal]:
        if not isinstance(data, str):
            return []

        signals: List[Signal] = []
        text = data.lower()

        if "error" in text:
            signals.append(Signal(
                id=f"text_error_{int(time.time()*1000)}",
                name="error_keyword",
                type="text",
                severity="low",
                confidence=0.6,
                data={"keyword": "error"},
            ))

        return signals

    def _generate_sql_destructive_signals(self, data: Any) -> List[Signal]:
        if not isinstance(data, str):
            return []

        sql = re.sub(r"\s+", " ", data.strip()).lower()
        signals: List[Signal] = []

        delete_match = re.search(r"delete\s+from\s+([a-zA-Z0-9_.]+)", sql)

        if delete_match and " where " not in sql:
            table = delete_match.group(1)
            signals.append(Signal(
                id=f"sql_delete_{table}_{int(time.time()*1000)}",
                name="destructive_sql",
                type="sql",
                severity="critical",
                confidence=0.95,
                data={
                    "operation": "DELETE",
                    "table": table,
                    "where_clause": False,
                },
                metadata={
                    "explain": "DELETE without WHERE removes all rows",
                },
            ))

        if sql.startswith("drop table"):
            signals.append(Signal(
                id=f"sql_drop_{int(time.time()*1000)}",
                name="drop_table",
                type="sql",
                severity="critical",
                confidence=0.99,
                data={"operation": "DROP TABLE"},
            ))

        return signals

    def _generate_pgbench_signals(self, data: Any) -> List[Signal]:
        """
        Generate signals from pgbench / JSON metrics.
        """
        if not isinstance(data, dict):
            return []

        signals: List[Signal] = []
        now = int(time.time() * 1000)

        tps = data.get("tps")
        latency = data.get("latency_ms", {}).get("avg_ms")
        errors = data.get("errors", 0)

        if tps is not None and tps < PG_LOW_TPS:
            signals.append(Signal(
                id=f"pgbench_low_tps_{now}",
                name="low_tps",
                type="pgbench",
                severity="high",
                confidence=0.9,
                data={"tps": tps},
                metadata={
                    "explain": "Low transactions per second detected",
                },
            ))

        if latency is not None and latency > PG_HIGH_LATENCY_MS:
            signals.append(Signal(
                id=f"pgbench_high_latency_{now}",
                name="high_latency",
                type="pgbench",
                severity="high",
                confidence=0.92,
                data={"avg_latency_ms": latency},
                metadata={
                    "explain": "High latency indicates slow queries or I/O bottlenecks",
                },
            ))

        if errors and errors > 0:
            signals.append(Signal(
                id=f"pgbench_errors_{now}",
                name="pgbench_errors",
                type="pgbench",
                severity="critical",
                confidence=0.98,
                data={"error_count": errors},
                metadata={
                    "explain": "Errors occurred during pgbench execution",
                },
            ))

        return signals

    def _generate_json_context_signals(self, data: Any) -> List[Signal]:
        """
        Generate signals from JSON context (observations, log_evidence, etc.)

        Detects:
        - Missing indexes on large tables
        - Large bulk operations
        - Sequential scan indicators
        - Vacuum/Analyze activity
        """
        if not isinstance(data, dict):
            self.logger.debug("_generate_json_context_signals: data is not a dict, type=%s", type(data))
            return []

        signals: List[Signal] = []
        now = int(time.time() * 1000)
        
        # DEBUG: Log the data structure
        self.logger.debug("_generate_json_context_signals: data keys=%s", list(data.keys()))
        
        context = data.get("context", {})
        observations = context.get("observations", [])
        log_evidence = context.get("log_evidence", [])
        
        # DEBUG: Log context extraction
        self.logger.debug("context=%s", context)
        self.logger.debug("observations=%s", observations)
        self.logger.debug("log_evidence=%s", log_evidence)

        # 1. Detect missing indexes on large tables
        rows_inserted = context.get("rows_inserted", 0)
        row_count = context.get("row_count", 0)
        table = context.get("table", "unknown")

        if (rows_inserted > 100000 or row_count > 100000):
            has_no_index_evidence = any(
                "no index" in str(evidence).lower() or "index scans: 0" in str(evidence).lower()
                for evidence in log_evidence
            )
            has_large_table_obs = any(
                "no index" in str(obs).lower() or "without index" in str(obs).lower()
                for obs in observations
            )

            if has_no_index_evidence or has_large_table_obs:
                signals.append(Signal(
                    id=f"missing_index_{table}_{now}",
                    name="missing_index_large_table",
                    type="json_context",
                    severity="high",
                    confidence=0.92,
                    data={
                        "table": table,
                        "row_count": rows_inserted or row_count,
                        "issue": "Large table without indexes causes sequential scans",
                    },
                    metadata={
                        "explain": f"Table '{table}' has {rows_inserted or row_count} rows with no indexes. SELECT queries will require sequential scans.",
                    },
                ))

        # 2. Detect large bulk operations
        if rows_inserted > 50000:
            signals.append(Signal(
                id=f"bulk_insert_{table}_{now}",
                name="large_bulk_insert",
                type="json_context",
                severity="medium",
                confidence=0.88,
                data={
                    "operation": "bulk_insert",
                    "table": table,
                    "rows": rows_inserted,
                },
                metadata={
                    "explain": f"Bulk insert of {rows_inserted} rows detected. Monitor for autovacuum needs.",
                },
            ))

        # 3. Detect vacuum/analyze activity (maintenance operations)
        vacuum_detected = any(
            "vacuum" in str(evidence).lower() or "analyze" in str(evidence).lower()
            for evidence in log_evidence
        )
        if vacuum_detected:
            signals.append(Signal(
                id=f"vacuum_analyze_{table}_{now}",
                name="maintenance_operation",
                type="json_context",
                severity="low",
                confidence=0.95,
                data={
                    "operation": "vacuum_or_analyze",
                    "table": table,
                },
                metadata={
                    "explain": "Automatic maintenance operation detected. PostgreSQL is reclaiming space/updating statistics.",
                },
            ))

        # 4. Detect sequential scan indicators
        seq_scan_evidence = any(
            "seq scan" in str(evidence).lower() or "sequential" in str(evidence).lower()
            for evidence in log_evidence
        )
        if seq_scan_evidence:
            signals.append(Signal(
                id=f"sequential_scan_{table}_{now}",
                name="sequential_scan_detected",
                type="json_context",
                severity="high",
                confidence=0.90,
                data={
                    "table": table,
                    "issue": "Sequential scan detected on large table",
                },
                metadata={
                    "explain": "Sequential scans on large tables are inefficient. Index creation is recommended.",
                },
            ))

        # 5. Detect CREATE INDEX actions (fix applied)
        action_taken = context.get("action_taken", [])
        create_index_actions = [a for a in action_taken if "CREATE INDEX" in str(a).upper()]
        if create_index_actions:
            for action in create_index_actions:
                # Extract index name
                match = re.search(r'CREATE INDEX\s+(?:CONCURRENTLY\s+)?(\S+)', action, re.IGNORECASE)
                index_name = match.group(1) if match else "unknown"

                signals.append(Signal(
                    id=f"index_created_{index_name}_{now}",
                    name="index_fix_applied",
                    type="json_context",
                    severity="low",
                    confidence=0.95,
                    data={
                        "index_name": index_name,
                        "table": table,
                        "action": action,
                    },
                    metadata={
                        "explain": f"Index '{index_name}' was created on '{table}'. Query performance should improve.",
                    },
                ))

        return signals

    # -------------------------------------------------------------------
    # Analyzers
    # -------------------------------------------------------------------

    def _analyze_metadata(self, signal: Signal, data: Any) -> None:
        signal.metadata.update({
            "analyzed_at": datetime.utcnow().isoformat(),
            "engine": "signal_engine_v4",
        })

    # ------------------------------------------------------------------
    # Incident/Alert Signal Generator
    # ------------------------------------------------------------------

    def _generate_incident_signals(self, data: Any) -> List[Signal]:
        """
        Generate signals from incident/alert JSON format.
        
        Detects:
        - P1/P2 incidents
        - High deviation factors
        - Query performance issues
        - Blocking detected
        """
        if not isinstance(data, dict):
            return []

        signals: List[Signal] = []
        now = int(time.time() * 1000)

        # Check for incident_metadata
        incident = data.get("incident_metadata", {})
        if incident:
            severity = incident.get("severity", "")
            incident_type = incident.get("incident_type", "")
            status = incident.get("status", "")

            if severity in ["P1", "P2"] or status == "ongoing":
                signals.append(Signal(
                    id=f"incident_{incident.get('incident_id', now)}",
                    name="active_incident",
                    type="incident",
                    severity="critical" if severity == "P1" else "high",
                    confidence=0.95,
                    data={
                        "incident_id": incident.get("incident_id"),
                        "incident_type": incident_type,
                        "severity": severity,
                        "status": status,
                    },
                    metadata={
                        "explain": f"Active {severity} incident detected: {incident_type}",
                    },
                ))

        # Check for alert
        alert = data.get("alert", {})
        if alert:
            alert_body = alert.get("alert_body", {})
            deviation = alert_body.get("deviation_factor", 0)
            
            if deviation >= 10:
                signals.append(Signal(
                    id=f"alert_high_deviation_{now}",
                    name="query_performance_alert",
                    type="incident",
                    severity="critical",
                    confidence=0.95,
                    data={
                        "query_id": alert_body.get("query_id"),
                        "deviation_factor": deviation,
                        "expected_ms": alert_body.get("expected_runtime_ms"),
                        "actual_ms": alert_body.get("current_runtime_ms"),
                    },
                    metadata={
                        "explain": f"Query execution time is {deviation}x higher than baseline",
                    },
                ))

        # Check for locking_analysis
        locking = data.get("locking_analysis", {})
        if locking.get("blocking_detected"):
            signals.append(Signal(
                id=f"blocking_detected_{now}",
                name="blocking_detected",
                type="incident",
                severity="high",
                confidence=0.95,
                data={
                    "wait_event_type": locking.get("wait_event_type"),
                    "reason": locking.get("reasoning"),
                },
                metadata={
                    "explain": "Blocking transaction detected causing query delays",
                },
            ))

        return signals

    # ------------------------------------------------------------------
    # Index Health Signal Generator
    # ------------------------------------------------------------------

    def _generate_index_health_signals(self, data: Any) -> List[Signal]:
        """
        Generate signals from index health JSON format.
        
        Detects:
        - Unused indexes
        - High bloat
        - Low usage ratios
        - Overlapping indexes
        """
        if not isinstance(data, dict):
            return []

        signals: List[Signal] = []
        now = int(time.time() * 1000)

        index_health = data.get("index_health", {})
        if not index_health:
            return signals

        # Check cross_index_analysis
        cross_analysis = index_health.get("cross_index_analysis", {})
        health_score = cross_analysis.get("overall_index_health_score", 100)

        if health_score < 70:
            signals.append(Signal(
                id=f"index_health_score_{now}",
                name="poor_index_health",
                type="index_health",
                severity="high",
                confidence=0.90,
                data={
                    "health_score": health_score,
                    "unused_indexes": cross_analysis.get("unused_index_candidates", []),
                    "high_bloat_indexes": cross_analysis.get("high_bloat_indexes", []),
                },
                metadata={
                    "explain": f"Index health score is {health_score}/100 - investigation needed",
                },
            ))

        # Check index_inventory for individual issues
        for idx in index_health.get("index_inventory", []):
            table_name = idx.get("table_name", "unknown")
            index_name = idx.get("index_name", "unknown")
            usage_ratio = idx.get("usage_metrics", {}).get("usage_ratio", 1.0)
            bloat_severity = idx.get("bloat_analysis", {}).get("bloat_severity", "low")
            risk_level = idx.get("risk_assessment", {}).get("current_risk_level", "low")

            # Unused or very low usage index
            if usage_ratio < 0.1:
                signals.append(Signal(
                    id=f"unused_index_{index_name}_{now}",
                    name="unused_index",
                    type="index_health",
                    severity="medium",
                    confidence=0.85,
                    data={
                        "index_name": index_name,
                        "table_name": table_name,
                        "usage_ratio": usage_ratio,
                    },
                    metadata={
                        "explain": f"Index '{index_name}' has very low usage ({usage_ratio:.2%})",
                    },
                ))

            # High bloat
            if bloat_severity == "high":
                bloat_percent = idx.get("bloat_analysis", {}).get("estimated_bloat_percent", 0)
                signals.append(Signal(
                    id=f"high_bloat_{index_name}_{now}",
                    name="high_index_bloat",
                    type="index_health",
                    severity="medium",
                    confidence=0.88,
                    data={
                        "index_name": index_name,
                        "table_name": table_name,
                        "bloat_percent": bloat_percent,
                    },
                    metadata={
                        "explain": f"Index '{index_name}' has {bloat_percent}% bloat",
                    },
                ))

            # High risk
            if risk_level == "high":
                risks = idx.get("risk_assessment", {}).get("primary_risks", [])
                signals.append(Signal(
                    id=f"high_risk_index_{index_name}_{now}",
                    name="high_risk_index",
                    type="index_health",
                    severity="high",
                    confidence=0.90,
                    data={
                        "index_name": index_name,
                        "table_name": table_name,
                        "risks": risks,
                    },
                    metadata={
                        "explain": f"Index '{index_name}' has high risk level",
                    },
                ))

        return signals

    # ------------------------------------------------------------------
    # Query Metrics Signal Generator
    # ------------------------------------------------------------------

    def _generate_query_metrics_signals(self, data: Any) -> List[Signal]:
        """
        Generate signals from query metrics JSON format.
        
        Detects:
        - High latency queries
        - High temp file usage
        - Low cache hit ratios
        - Plan changes
        """
        if not isinstance(data, dict):
            return []

        signals: List[Signal] = []
        now = int(time.time() * 1000)

        query_metrics = data.get("query_metrics", {})
        if not query_metrics:
            return signals

        # Check for derived_metrics first (has deviation factor)
        derived = query_metrics.get("derived_metrics", {})
        deviation = derived.get("latency_deviation_factor", 0)
        cache_hit = derived.get("cache_hit_ratio", 1.0)
        temp_spill = derived.get("temp_spill_detected", False)

        # High deviation factor
        if deviation >= 10:
            signals.append(Signal(
                id=f"query_deviation_{now}",
                name="high_query_latency_deviation",
                type="query_metrics",
                severity="critical",
                confidence=0.95,
                data={
                    "deviation_factor": deviation,
                    "latency_increase": f"{deviation}x baseline",
                },
                metadata={
                    "explain": f"Query execution time is {deviation}x higher than baseline",
                },
            ))

        # Low cache hit ratio
        if cache_hit < 0.90:
            signals.append(Signal(
                id=f"low_cache_hit_{now}",
                name="low_buffer_cache_hit_ratio",
                type="query_metrics",
                severity="medium",
                confidence=0.85,
                data={
                    "cache_hit_ratio": cache_hit,
                },
                metadata={
                    "explain": f"Buffer cache hit ratio is {cache_hit:.1%} - may need more shared_buffers",
                },
            ))

        # Temp spill detected
        if temp_spill:
            signals.append(Signal(
                id=f"temp_spill_{now}",
                name="temp_file_spill_detected",
                type="query_metrics",
                severity="medium",
                confidence=0.88,
                data={
                    "temp_spill_detected": True,
                },
                metadata={
                    "explain": "Query spilled to temp files - work_mem may be insufficient",
                },
            ))

        # Check pg_stat_statements_metrics
        pg_stats = query_metrics.get("pg_stat_statements_metrics", {})
        if pg_stats:
            mean_time = pg_stats.get("mean_time_ms", 0)
            temp_blocks = pg_stats.get("temp_blks_written", 0)

            # High mean time
            if mean_time > 5000:
                signals.append(Signal(
                    id=f"high_query_latency_{now}",
                    name="high_query_latency",
                    type="query_metrics",
                    severity="high",
                    confidence=0.92,
                    data={
                        "mean_time_ms": mean_time,
                        "total_time_ms": pg_stats.get("total_time_ms", 0),
                        "calls": pg_stats.get("calls", 0),
                    },
                    metadata={
                        "explain": f"Query mean execution time is {mean_time}ms - very high",
                    },
                ))

            # High temp blocks
            if temp_blocks > 10000:
                signals.append(Signal(
                    id=f"high_temp_usage_{now}",
                    name="high_temp_file_usage",
                    type="query_metrics",
                    severity="medium",
                    confidence=0.88,
                    data={
                        "temp_blocks_written": temp_blocks,
                    },
                    metadata={
                        "explain": f"Query wrote {temp_blocks} temp blocks - memory/work_mem may be insufficient",
                    },
                ))

        # Check for planner_executor_behavior (sequential scan heavy)
        planner = query_metrics.get("planner_executor_behavior", {})
        scan_methods = planner.get("scan_methods", {})
        seq_scan = scan_methods.get("sequential_scan", {})
        if seq_scan.get("detected") and seq_scan.get("frequency_percent", 0) > 50:
            signals.append(Signal(
                id=f"sequential_scan_{now}",
                name="sequential_scan_heavy",
                type="query_metrics",
                severity="high",
                confidence=0.88,
                data={
                    "seq_scan_percent": seq_scan.get("frequency_percent"),
                    "tables": seq_scan.get("tables", []),
                },
                metadata={
                    "explain": f"Sequential scan detected {seq_scan.get('frequency_percent')}% of the time",
                },
            ))

        # Check for row_estimation_anomaly
        row_est = query_metrics.get("row_estimation_accuracy", {})
        error_factor = row_est.get("estimation_error_factor", 0)
        if error_factor > 100:
            signals.append(Signal(
                id=f"row_estimation_error_{now}",
                name="severe_row_estimation_error",
                type="query_metrics",
                severity="high",
                confidence=0.90,
                data={
                    "estimated_rows": row_est.get("estimated_rows"),
                    "actual_rows": row_est.get("actual_rows"),
                    "error_factor": error_factor,
                },
                metadata={
                    "explain": f"Row estimation off by {error_factor}x - statistics may be stale",
                },
            ))

        # Check for plan changes
        plan_stability = query_metrics.get("query_plan_stability", {})
        if plan_stability.get("plan_changed"):
            reasons = plan_stability.get("plan_change_reason", [])
            signals.append(Signal(
                id=f"plan_change_{now}",
                name="query_plan_regression",
                type="query_metrics",
                severity="medium",
                confidence=0.85,
                data={
                    "current_hash": plan_stability.get("current_plan_hash"),
                    "baseline_hash": plan_stability.get("baseline_plan_hash"),
                    "reasons": reasons,
                },
                metadata={
                    "explain": "Query plan has changed from baseline",
                },
            ))

        return signals

    # ------------------------------------------------------------------
    # Table Statistics Signal Generator
    # ------------------------------------------------------------------

    def _generate_table_stats_signals(self, data: Any) -> List[Signal]:
        """
        Generate signals from table statistics JSON format.
        
        Detects:
        - High dead tuples
        - Stale statistics
        - High sequential scans
        """
        if not isinstance(data, dict):
            return []

        signals: List[Signal] = []
        now = int(time.time() * 1000)

        table_stats = data.get("table_statistics", {})
        if not table_stats:
            return signals

        tables = table_stats.get("tables", [])
        diagnostic_hints = table_stats.get("diagnostic_hints", [])

        for table in tables:
            table_name = table.get("table_name", "unknown")
            dead_tuples = table.get("n_dead_tup", 0)
            seq_scans = table.get("seq_scan", 0)
            idx_scans = table.get("idx_scan", 0)
            last_analyze = table.get("last_analyze", "")
            last_vacuum = table.get("last_vacuum", "")

            # High dead tuples
            if dead_tuples > 100000:
                signals.append(Signal(
                    id=f"high_dead_tuples_{table_name}_{now}",
                    name="high_dead_tuples",
                    type="table_stats",
                    severity="medium",
                    confidence=0.90,
                    data={
                        "table_name": table_name,
                        "dead_tuples": dead_tuples,
                    },
                    metadata={
                        "explain": f"Table '{table_name}' has {dead_tuples:,} dead tuples - VACUUM recommended",
                    },
                ))

            # Sequential scan heavy (more seq than idx scans)
            if seq_scans > idx_scans * 10 and seq_scans > 1000:
                signals.append(Signal(
                    id=f"seq_scan_heavy_{table_name}_{now}",
                    name="sequential_scan_heavy",
                    type="table_stats",
                    severity="high",
                    confidence=0.88,
                    data={
                        "table_name": table_name,
                        "seq_scans": seq_scans,
                        "idx_scans": idx_scans,
                    },
                    metadata={
                        "explain": f"Table '{table_name}' has {seq_scans:,} seq scans vs {idx_scans:,} idx scans",
                    },
                ))

        # Check for stale statistics
        stale_hints = [h for h in diagnostic_hints if "stale" in h.lower() or "dead" in h.lower()]
        if stale_hints:
            signals.append(Signal(
                id=f"stale_statistics_{now}",
                name="stale_table_statistics",
                type="table_stats",
                severity="medium",
                confidence=0.85,
                data={
                    "hints": stale_hints,
                },
                    metadata={
                        "explain": "Table statistics are stale - ANALYZE recommended",
                    },
                ))

        return signals

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _filter_low_confidence(self, signals: List[Signal]) -> List[Signal]:
        return [s for s in signals if s.confidence >= 0.6]
