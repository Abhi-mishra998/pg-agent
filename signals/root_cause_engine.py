#!/usr/bin/env python3
"""
RootCauseEngine - Deterministic Root Cause Analysis for PostgreSQL

Design: Rule-based engine that maps signals/evidence to root causes.
Integration: Post-processor pattern (does NOT modify SignalEngine).

Root Cause Categories:
1. Index Issues
2. Statistics / Maintenance
3. Blocking / Locking
4. Configuration Parameters
5. Application Behavior
6. Capacity / Hardware
7. Background Jobs / Maintenance Tasks
8. Deployment or Schema Changes

Each rule specifies:
- required_evidence: What signals/types are needed
- confidence_contribution: How much each piece adds to confidence
- common false positives: Known scenarios that trigger false positives
- Engine supports multiple root causes per incident
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
from collections import defaultdict

# Import RootCauseCategory from core/enums for consistency
from core.enums import RootCauseCategory


class EvidenceType(Enum):
    """Types of evidence from SignalEngine."""
    SEQUENTIAL_SCAN = "sequential_scan_detected"
    MISSING_INDEX = "missing_index_large_table"
    UNUSED_INDEX = "unused_index"
    HIGH_BLOAT = "high_index_bloat"
    STALE_STATS = "stale_statistics"
    HIGH_DEAD_TUPLES = "high_dead_tuples"
    BLOCKING = "blocking_detected"
    LOCK_WAIT = "lock_wait"
    HIGH_LATENCY = "high_query_latency"
    LOW_CACHE_HIT = "low_buffer_cache_hit_ratio"
    TEMP_SPILL = "temp_file_spill_detected"
    PLAN_CHANGE = "query_plan_regression"
    ROW_ESTIMATION_ERROR = "severe_row_estimation_error"
    LOW_TPS = "low_tps"
    HIGH_ERRORS = "pgbench_errors"
    MAINTENANCE_OPERATION = "maintenance_operation"


@dataclass
class RuleMatch:
    """Result of matching a rule against evidence."""
    rule_name: str
    category: RootCauseCategory
    matched_evidence: List[str]  # Evidence IDs
    confidence_contribution: float
    is_triggered: bool = True
    false_positive_warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RootCauseResult:
    """Result of root cause analysis for a single category."""
    category: RootCauseCategory
    is_likely_cause: bool
    confidence: float
    matched_rules: List[RuleMatch]
    evidence_ids: List[str] = field(default_factory=list)
    contributing_factors: List[str] = field(default_factory=list)
    causation_chain: List[str] = field(default_factory=list)
    false_positive_notes: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "category": self.category.value,
            "is_likely_cause": self.is_likely_cause,
            "confidence": self.confidence,
            "matched_rules": [
                {
                    "rule_name": m.rule_name,
                    "confidence_contribution": m.confidence_contribution,
                    "matched_evidence": m.matched_evidence,
                    "false_positive_warnings": m.false_positive_warnings,
                }
                for m in self.matched_rules
            ],
            "evidence_ids": self.evidence_ids,
            "contributing_factors": self.contributing_factors,
            "causation_chain": self.causation_chain,
            "false_positive_notes": self.false_positive_notes,
            "recommendations": self.recommendations,
        }


# ---------------------------------------------------------------------
# Evidence Matcher
# ---------------------------------------------------------------------

@dataclass
class EvidenceSet:
    """Normalized evidence set for rule matching."""
    signal_ids: Set[str] = field(default_factory=set)
    signal_types: Set[str] = field(default_factory=set)
    signal_names: Set[str] = field(default_factory=set)
    metric_values: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    evidence_map: Dict[str, Any] = field(default_factory=dict)
    
    def has_signal_type(self, signal_type: str) -> bool:
        return signal_type in self.signal_types
    
    def has_signal_name(self, signal_name: str) -> bool:
        return signal_name in self.signal_names
    
    def get_metric(self, metric_name: str, default: float = 0.0) -> float:
        return self.metric_values.get(metric_name, default)
    
    def has_metric_above(self, metric_name: str, threshold: float) -> bool:
        return self.metric_values.get(metric_name, 0) > threshold
    
    def has_metric_below(self, metric_name: str, threshold: float) -> bool:
        return self.metric_values.get(metric_name, float('inf')) < threshold


# ---------------------------------------------------------------------
# Root Cause Rules
# ---------------------------------------------------------------------

@dataclass
class RootCauseRule:
    """
    A deterministic rule for root cause identification.
    
    Each rule specifies:
    - category: Which root cause category this rule addresses
    - required_evidence: Signal types/names that must be present
    - optional_evidence: Signal types that strengthen confidence
    - confidence_contribution: Base confidence when rule matches
    - max_confidence: Maximum confidence this rule can contribute
    - false_positives: Known scenarios that cause false positives
    - contributing_factors: Factors to add when rule triggers
    - causation_steps: Step-by-step causation chain
    - recommendations: Suggested actions
    """
    name: str
    category: RootCauseCategory
    required_evidence: List[str]  # Signal types that MUST be present
    optional_evidence: List[str] = field(default_factory=list)
    confidence_contribution: float = 0.3
    max_confidence: float = 0.95
    false_positives: List[str] = field(default_factory=list)
    contributing_factors: List[str] = field(default_factory=list)
    causation_steps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def matches(self, evidence: EvidenceSet) -> Tuple[bool, List[str], List[str]]:
        """
        Check if rule matches evidence.
        Returns: (is_match, matched_evidence_ids, false_positive_warnings)
        """
        matched = []
        warnings = []
        
        # Check required evidence
        for req in self.required_evidence:
            if evidence.has_signal_type(req):
                matched.append(req)
            elif evidence.has_signal_name(req):
                matched.append(req)
            else:
                return False, [], []  # Required evidence not found
        
        # Check optional evidence (for confidence boost)
        for opt in self.optional_evidence:
            if evidence.has_signal_type(opt) or evidence.has_signal_name(opt):
                matched.append(opt)
        
        # Check for false positives
        for fp in self.false_positives:
            if self._check_false_positive(fp, evidence):
                warnings.append(fp)
        
        return True, matched, warnings
    
    def _check_false_positive(self, fp: str, evidence: EvidenceSet) -> bool:
        """Check if a false positive condition is met."""
        # Format: "condition:param:value"
        if fp.startswith("metric_below:"):
            _, metric, threshold = fp.split(":")
            return evidence.has_metric_below(metric, float(threshold))
        elif fp.startswith("metric_above:"):
            _, metric, threshold = fp.split(":")
            return evidence.has_metric_above(metric, float(threshold))
        elif fp == "small_table":
            row_count = evidence.get_metric("row_count", float('inf'))
            return row_count < 10000
        elif fp == "recent_vacuum":
            vacuum_time = evidence.get_metric("hours_since_vacuum", 0)
            return vacuum_time < 1
        elif fp == "maintenance_window":
            return evidence.metadata.get("maintenance_window", False)
        return False


# ---------------------------------------------------------------------
# Rule Registry
# ---------------------------------------------------------------------

class RootCauseRuleRegistry:
    """
    Registry of all root cause detection rules.
    
    These are deterministic, expert-defined rules based on PostgreSQL
    performance best practices and common troubleshooting patterns.
    """
    
    def __init__(self):
        self.rules: List[RootCauseRule] = []
        self._register_all_rules()
    
    def _register_all_rules(self) -> None:
        """Register all root cause detection rules."""
        
        # 1. INDEX ISSUES
        self.rules.append(RootCauseRule(
            name="sequential_scan_large_table",
            category=RootCauseCategory.INDEX_ISSUES,
            required_evidence=["sequential_scan_detected"],
            optional_evidence=["high_query_latency", "missing_index_large_table"],
            confidence_contribution=0.35,
            max_confidence=0.90,
            false_positives=["small_table", "SELECT count(*) queries"],
            contributing_factors=[
                "Table scan is accessing majority of rows",
                "No index available for filter predicates",
            ],
            causation_steps=[
                "Query lacks index on filter columns",
                "PostgreSQL chooses sequential scan",
                "Scan must read all pages from disk/memory",
                "Query time increases linearly with table size",
            ],
            recommendations=[
                "CREATE INDEX on filter columns",
                "Consider composite index for multi-column filters",
            ],
        ))
        
        self.rules.append(RootCauseRule(
            name="missing_index_large_table",
            category=RootCauseCategory.INDEX_ISSUES,
            required_evidence=["missing_index_large_table"],
            optional_evidence=["sequential_scan_detected", "high_query_latency"],
            confidence_contribution=0.40,
            max_confidence=0.95,
            false_positives=["small_table", "bulk_load_operation"],
            contributing_factors=[
                "Large table (>100K rows) without appropriate index",
                "Query pattern targets unindexed columns",
            ],
            causation_steps=[
                "Application queries large table without index",
                "Planner has no index option available",
                "Sequential scan is forced",
            ],
            recommendations=[
                "CREATE INDEX CONCURRENTLY on target columns",
                "Use INCLUDE columns for covering indexes",
            ],
        ))
        
        self.rules.append(RootCauseRule(
            name="unused_index_candidate",
            category=RootCauseCategory.INDEX_ISSUES,
            required_evidence=["unused_index"],
            optional_evidence=["high_index_bloat"],
            confidence_contribution=0.25,
            max_confidence=0.80,
            false_positives=["recent_deployment", "conditional_queries"],
            contributing_factors=[
                "Index exists but query patterns don't use it",
            ],
            recommendations=[
                "Review index usage with pg_stat_user_indexes",
                "Consider dropping unused index",
            ],
        ))
        
        self.rules.append(RootCauseRule(
            name="high_index_bloat",
            category=RootCauseCategory.INDEX_ISSUES,
            required_evidence=["high_index_bloat"],
            optional_evidence=["maintenance_operation"],
            confidence_contribution=0.30,
            max_confidence=0.85,
            false_positives=["recent_vacuum", "high_write_table"],
            contributing_factors=[
                "Index has excessive bloat (>50%)",
                "Vacuum hasn't reclaimed space",
            ],
            causation_steps=[
                "Heavy UPDATEs/DELETEs cause index tuple fragmentation",
                "Bloat increases index size and slows scans",
            ],
            recommendations=["REINDEX TABLE", "Ensure regular VACUUM scheduling"],
        ))
        
        # 2. STATISTICS / MAINTENANCE
        self.rules.append(RootCauseRule(
            name="stale_statistics",
            category=RootCauseCategory.STATISTICS_MAINTENANCE,
            required_evidence=["stale_statistics"],
            optional_evidence=["plan_change", "severe_row_estimation_error"],
            confidence_contribution=0.35,
            max_confidence=0.90,
            false_positives=["recent_bulk_load", "stable_table"],
            contributing_factors=[
                "Statistics older than 7 days",
                "Table has had significant data changes (>10%)",
            ],
            causation_steps=[
                "Table statistics are stale",
                "Planner uses outdated cardinality estimates",
                "Wrong plan choice leads to suboptimal execution",
            ],
            recommendations=["Run ANALYZE on affected table"],
        ))
        
        self.rules.append(RootCauseRule(
            name="high_dead_tuples",
            category=RootCauseCategory.STATISTICS_MAINTENANCE,
            required_evidence=["high_dead_tuples"],
            optional_evidence=["sequential_scan_heavy", "maintenance_operation"],
            confidence_contribution=0.30,
            max_confidence=0.85,
            false_positives=["high_write_table", "recent_vacuum"],
            contributing_factors=[
                "Dead tuple ratio > 20%",
                "Vacuum not keeping pace with deletes/updates",
            ],
            causation_steps=[
                "UPDATEs/DELETEs leave dead tuples",
                "High dead tuple ratio wastes buffer cache",
            ],
            recommendations=["Run VACUUM ANALYZE on table"],
        ))
        
        self.rules.append(RootCauseRule(
            name="row_estimation_error",
            category=RootCauseCategory.STATISTICS_MAINTENANCE,
            required_evidence=["severe_row_estimation_error"],
            optional_evidence=["sequential_scan_detected", "high_query_latency"],
            confidence_contribution=0.40,
            max_confidence=0.92,
            false_positives=["skewed_data", "correlated_columns"],
            contributing_factors=[
                "Planner estimates differ significantly from actual rows",
            ],
            causation_steps=[
                "Statistics are inaccurate",
                "Planner misestimates row counts",
                "Wrong join type or scan method chosen",
            ],
            recommendations=[
                "Run ANALYZE to update statistics",
                "Use extended statistics for correlated columns",
            ],
        ))
        
        # 3. BLOCKING / LOCKING
        self.rules.append(RootCauseRule(
            name="blocking_detected",
            category=RootCauseCategory.BLOCKING_LOCKING,
            required_evidence=["blocking_detected"],
            optional_evidence=["lock_wait", "high_query_latency"],
            confidence_contribution=0.45,
            max_confidence=0.95,
            false_positives=["normal_short_locks", "maintenance_window"],
            contributing_factors=[
                "Long-running transaction holding lock",
                "Concurrent transaction waiting for lock",
            ],
            causation_steps=[
                "Transaction A acquires lock",
                "Transaction B requests same lock",
                "Transaction B is blocked",
            ],
            recommendations=[
                "Identify blocking transaction with pg_blocking_pids",
                "Consider shorter transactions",
            ],
        ))
        
        self.rules.append(RootCauseRule(
            name="lock_wait_timeout",
            category=RootCauseCategory.BLOCKING_LOCKING,
            required_evidence=["lock_wait"],
            optional_evidence=["blocking_detected", "high_query_latency"],
            confidence_contribution=0.35,
            max_confidence=0.88,
            false_positives=["high_lock_table", "serializable_isolation"],
            contributing_factors=[
                "Lock wait exceeded threshold",
            ],
            causation_steps=[
                "Session attempts to acquire lock",
                "Lock held by other session",
                "Wait exceeds lock_timeout",
            ],
            recommendations=["Check lock_timeout setting", "Identify lock holders"],
        ))
        
        # 4. CONFIGURATION PARAMETERS
        self.rules.append(RootCauseRule(
            name="low_shared_buffers",
            category=RootCauseCategory.CONFIGURATION,
            required_evidence=["low_buffer_cache_hit_ratio"],
            optional_evidence=["high_query_latency", "sequential_scan_heavy"],
            confidence_contribution=0.35,
            max_confidence=0.85,
            false_positives=["large_data_scan", "cold_cache"],
            contributing_factors=[
                "shared_buffers too small for working set",
            ],
            causation_steps=[
                "Working set exceeds shared_buffers",
                "Pages evicted frequently",
            ],
            recommendations=["Increase shared_buffers (up to 40% of RAM)"],
        ))
        
        self.rules.append(RootCauseRule(
            name="insufficient_work_mem",
            category=RootCauseCategory.CONFIGURATION,
            required_evidence=["temp_file_spill_detected"],
            optional_evidence=["high_query_latency"],
            confidence_contribution=0.35,
            max_confidence=0.88,
            false_positives=["hash_aggregates", "very_large_result"],
            contributing_factors=[
                "work_mem too small for query operations",
            ],
            causation_steps=[
                "Query requires more memory than work_mem",
                "Operations spill to disk",
            ],
            recommendations=["Increase work_mem (per operation limit)"],
        ))
        
        self.rules.append(RootCauseRule(
            name="plan_regression",
            category=RootCauseCategory.CONFIGURATION,
            required_evidence=["query_plan_regression"],
            optional_evidence=["high_query_latency", "plan_change"],
            confidence_contribution=0.40,
            max_confidence=0.90,
            false_positives=["statistics_update", "parameterized_plans"],
            contributing_factors=[
                "Query plan changed from baseline",
            ],
            causation_steps=[
                "Plan hash changed",
                "New plan has different characteristics",
                "Execution time increased",
            ],
            recommendations=["Review plan differences with EXPLAIN"],
        ))
        
        # 5. APPLICATION BEHAVIOR
        self.rules.append(RootCauseRule(
            name="n_plus_one_queries",
            category=RootCauseCategory.APPLICATION_BEHAVIOR,
            required_evidence=["high_query_latency"],
            optional_evidence=["high_latency"],
            confidence_contribution=0.25,
            max_confidence=0.75,
            false_positives=["reporting_queries", "batch_processing"],
            contributing_factors=[
                "Application makes repeated similar queries",
            ],
            recommendations=["Batch queries using IN clauses"],
        ))
        
        self.rules.append(RootCauseRule(
            name="unbounded_result_set",
            category=RootCauseCategory.APPLICATION_BEHAVIOR,
            required_evidence=["high_query_latency"],
            optional_evidence=["sequential_scan_heavy"],
            confidence_contribution=0.30,
            max_confidence=0.82,
            false_positives=["export_operation", "admin_query"],
            contributing_factors=[
                "Query returns large result set",
                "No LIMIT clause applied",
            ],
            recommendations=["Add LIMIT clause", "Implement pagination"],
        ))
        
        # 6. CAPACITY / HARDWARE
        self.rules.append(RootCauseRule(
            name="disk_io_bottleneck",
            category=RootCauseCategory.CAPACITY_HARDWARE,
            required_evidence=["low_tps"],
            optional_evidence=["high_query_latency", "sequential_scan_heavy"],
            confidence_contribution=0.35,
            max_confidence=0.85,
            false_positives=["shared_storage", "network_storage"],
            contributing_factors=[
                "Disk I/O capacity saturated",
            ],
            causation_steps=[
                "I/O operations queue up",
                "Read/write latency increases",
            ],
            recommendations=["Monitor I/O metrics with iostat", "Consider faster storage"],
        ))
        
        self.rules.append(RootCauseRule(
            name="memory_pressure",
            category=RootCauseCategory.CAPACITY_HARDWARE,
            required_evidence=["low_buffer_cache_hit_ratio"],
            optional_evidence=["high_query_latency", "temp_file_spill_detected"],
            confidence_contribution=0.30,
            max_confidence=0.80,
            false_positives=["cold_cache", "benchmark_load"],
            contributing_factors=[
                "System memory pressure",
            ],
            recommendations=["Check overall system memory", "Consider memory upgrade"],
        ))
        
        self.rules.append(RootCauseRule(
            name="cpu_saturation",
            category=RootCauseCategory.CAPACITY_HARDWARE,
            required_evidence=["high_query_latency"],
            optional_evidence=["high_cpu_usage"],
            confidence_contribution=0.25,
            max_confidence=0.75,
            false_positives=["batch_job_running", "maintenance_operation"],
            contributing_factors=["CPU utilization at capacity"],
            recommendations=["Check CPU utilization", "Consider CPU upgrade"],
        ))
        
        # 7. BACKGROUND JOBS / MAINTENANCE TASKS
        self.rules.append(RootCauseRule(
            name="autovacuum_interference",
            category=RootCauseCategory.BACKGROUND_JOBS,
            required_evidence=["maintenance_operation"],
            optional_evidence=["high_query_latency", "low_tps"],
            confidence_contribution=0.30,
            max_confidence=0.80,
            false_positives=["maintenance_window", "idle_system"],
            contributing_factors=[
                "Autovacuum consuming resources",
            ],
            recommendations=["Review autovacuum settings"],
        ))
        
        self.rules.append(RootCauseRule(
            name="long_running_transaction",
            category=RootCauseCategory.BACKGROUND_JOBS,
            required_evidence=["blocking_detected"],
            optional_evidence=["high_dead_tuples"],
            confidence_contribution=0.35,
            max_confidence=0.88,
            false_positives=["report_generation", "batch_import"],
            contributing_factors=[
                "Transaction open for extended period",
            ],
            recommendations=["Keep transactions short", "Set statement_timeout"],
        ))
        
        # 8. DEPLOYMENT / SCHEMA CHANGES
        self.rules.append(RootCauseRule(
            name="recent_deployment",
            category=RootCauseCategory.DEPLOYMENT_SCHEMA,
            required_evidence=["plan_change", "query_plan_regression"],
            optional_evidence=["high_query_latency"],
            confidence_contribution=0.35,
            max_confidence=0.85,
            false_positives=["statistics_update", "data_distribution_change"],
            contributing_factors=[
                "Recent deployment changed query patterns",
            ],
            recommendations=["Review deployment changelog"],
        ))
        
        self.rules.append(RootCauseRule(
            name="schema_change_impact",
            category=RootCauseCategory.DEPLOYMENT_SCHEMA,
            required_evidence=["sequential_scan_detected"],
            optional_evidence=["plan_change"],
            confidence_contribution=0.30,
            max_confidence=0.80,
            false_positives=["new_query_type", "legitimate_data_access_pattern"],
            contributing_factors=[
                "Schema change removed or altered index",
            ],
            recommendations=["Review recent DDL changes"],
        ))
    
    def get_rules_for_category(self, category: RootCauseCategory) -> List[RootCauseRule]:
        return [r for r in self.rules if r.category == category]
    
    def get_all_categories(self) -> List[RootCauseCategory]:
        return list(set(r.category for r in self.rules))


# ---------------------------------------------------------------------
# Root Cause Engine
# ---------------------------------------------------------------------

class RootCauseEngine:
    """
    Deterministic root cause analysis engine.
    
    Takes signals from SignalEngine and produces root cause analysis.
    Integration: Post-processor pattern - does NOT modify SignalEngine.
    """
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self.registry = RootCauseRuleRegistry()
        self.min_confidence_threshold = 0.30
    
    def analyze(
        self,
        signal_result,
        evidence_collection=None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[RootCauseCategory, RootCauseResult]:
        """Analyze signals and identify root causes."""
        self.logger.info("Starting root cause analysis for %d signals", len(signal_result.signals))
        
        evidence_set = self._normalize_signals(signal_result, context)
        all_matches = self._match_all_rules(evidence_set)
        results = self._aggregate_by_category(all_matches)
        
        for category, result in results.items():
            result.causation_chain = self._build_causation_chain(category, result, evidence_set)
        
        self.logger.info("Root cause analysis complete: %d categories identified", len(results))
        return results
    
    def _normalize_signals(
        self,
        signal_result,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvidenceSet:
        evidence = EvidenceSet()
        
        for signal in signal_result.signals:
            evidence.signal_ids.add(signal.id)
            evidence.signal_types.add(signal.type)
            evidence.signal_names.add(signal.name)
            evidence.evidence_map[signal.id] = {
                "type": signal.type,
                "name": signal.name,
                "data": signal.data,
                "metadata": signal.metadata,
                "confidence": signal.confidence,
            }
            
            for key, value in signal.data.items():
                if isinstance(value, (int, float)):
                    evidence.metric_values[key] = value
            
            if "metric_value" in signal.metadata:
                evidence.metric_values[signal.metadata.get("metric_name", "unknown")] = \
                    signal.metadata["metric_value"]
        
        if context:
            evidence.metadata = context
        
        return evidence
    
    def _match_all_rules(self, evidence: EvidenceSet) -> List[RuleMatch]:
        matches = []
        
        for rule in self.registry.rules:
            is_triggered, matched_ids, warnings = rule.matches(evidence)
            
            if is_triggered:
                confidence = rule.confidence_contribution
                
                if warnings:
                    confidence *= (1.0 - (len(warnings) * 0.1))
                    confidence = max(0.1, confidence)
                
                match = RuleMatch(
                    rule_name=rule.name,
                    category=rule.category,
                    matched_evidence=matched_ids,
                    confidence_contribution=min(confidence, rule.max_confidence),
                    false_positive_warnings=warnings,
                    metadata={
                        "contributing_factors": rule.contributing_factors,
                        "causation_steps": rule.causation_steps,
                        "recommendations": rule.recommendations,
                    },
                )
                matches.append(match)
                self.logger.debug("Rule matched: %s (confidence: %.2f)", rule.name, confidence)
        
        return matches
    
    def _aggregate_by_category(
        self,
        matches: List[RuleMatch],
    ) -> Dict[RootCauseCategory, RootCauseResult]:
        category_matches: Dict[RootCauseCategory, List[RuleMatch]] = defaultdict(list)
        
        for match in matches:
            category_matches[match.category].append(match)
        
        results = {}
        
        for category, cat_matches in category_matches.items():
            total_confidence = sum(m.confidence_contribution for m in cat_matches)
            is_likely = total_confidence >= self.min_confidence_threshold
            
            evidence_ids = []
            for m in cat_matches:
                evidence_ids.extend(m.matched_evidence)
            evidence_ids = list(set(evidence_ids))
            
            contributing_factors = []
            false_positive_notes = []
            for m in cat_matches:
                contributing_factors.extend(m.metadata.get("contributing_factors", []))
                for fp in m.false_positive_warnings:
                    false_positive_notes.append(f"[{m.rule_name}] {fp}")
            
            recommendations = []
            for m in cat_matches:
                recommendations.extend(m.metadata.get("recommendations", []))
            recommendations = list(set(recommendations))
            
            results[category] = RootCauseResult(
                category=category,
                is_likely_cause=is_likely,
                confidence=round(min(total_confidence, 1.0), 2),
                matched_rules=cat_matches,
                evidence_ids=evidence_ids,
                contributing_factors=list(set(contributing_factors)),
                false_positive_notes=false_positive_notes,
                recommendations=recommendations,
            )
        
        return results
    
    def _build_causation_chain(
        self,
        category: RootCauseCategory,
        result: RootCauseResult,
        evidence: EvidenceSet,
    ) -> List[str]:
        if result.matched_rules:
            best_match = max(result.matched_rules, key=lambda m: m.confidence_contribution)
            return best_match.metadata.get("causation_steps", [])
        return []
    
    def get_primary_causes(
        self,
        results: Dict[RootCauseCategory, RootCauseResult],
        top_n: int = 3,
    ) -> List[Tuple[RootCauseCategory, RootCauseResult]]:
        likely_causes = [
            (cat, res) for cat, res in results.items()
            if res.is_likely_cause
        ]
        sorted_causes = sorted(
            likely_causes,
            key=lambda x: x[1].confidence,
            reverse=True,
        )
        return sorted_causes[:top_n]
    
    def format_results(
        self,
        results: Dict[RootCauseCategory, RootCauseResult],
    ) -> Dict[str, Any]:
        primary_causes = self.get_primary_causes(results)
        
        return {
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "categories_analyzed": len(results),
            "likely_causes_count": len([r for r in results.values() if r.is_likely_cause]),
            "primary_causes": [
                {
                    "category": cat.value,
                    "confidence": res.confidence,
                    "contributing_factors": res.contributing_factors[:3],
                    "recommendations": res.recommendations[:2],
                }
                for cat, res in primary_causes
            ],
            "all_results": {
                cat.value: res.to_dict()
                for cat, res in sorted(
                    results.items(),
                    key=lambda x: x[1].confidence,
                    reverse=True
                )
            },
        }

