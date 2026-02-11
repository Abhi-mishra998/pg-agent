#!/usr/bin/env python3
"""
Root Cause Engine Integration Example

Demonstrates how RootCauseEngine integrates with the existing pg-agent pipeline
without modifying SignalEngine.

This is a standalone example that can be run to verify the integration.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from signals.root_cause_engine import (
    RootCauseEngine,
    RootCauseCategory,
    EvidenceSet,
)
from signals.signal_engine import SignalEngine, Signal, SignalResult


# ---------------------------------------------------------------------
# Sample Data - Simulated Slow Query Incident
# ---------------------------------------------------------------------

def create_slow_query_incident_data() -> Dict[str, Any]:
    """Create sample data representing a slow query incident."""
    return {
        "table_statistics": {
            "tables": [
                {
                    "table_name": "orders_table",
                    "seq_scan": 1523,
                    "idx_scan": 12,
                    "n_dead_tup": 234567,
                    "last_analyze": "2026-01-18T03:00:00Z",
                    "last_vacuum": "2026-01-20T22:15:00Z",
                }
            ],
            "diagnostic_hints": ["Stale statistics detected - last ANALYZE 7 days ago"],
        },
        "query_metrics": {
            "derived_metrics": {
                "latency_deviation_factor": 30,
                "cache_hit_ratio": 0.65,
                "temp_spill_detected": False,
            },
            "pg_stat_statements_metrics": {
                "mean_time_ms": 15234.5,
                "total_time_ms": 22110000,
                "calls": 1452,
                "temp_blks_written": 0,
            },
            "planner_executor_behavior": {
                "scan_methods": {
                    "sequential_scan": {
                        "detected": True,
                        "frequency_percent": 99,
                        "tables": ["orders_table"],
                    }
                }
            },
            "row_estimation_accuracy": {
                "estimated_rows": 1000,
                "actual_rows": 356024,
                "estimation_error_factor": 356,
            },
            "query_plan_stability": {
                "plan_changed": True,
                "current_plan_hash": "xyz789",
                "baseline_plan_hash": "abc123",
                "plan_change_reason": ["statistics_updated"],
            },
        },
        "index_health": {
            "cross_index_analysis": {
                "overall_index_health_score": 45,
                "unused_index_candidates": [],
                "high_bloat_indexes": [],
            },
            "index_inventory": [
                {
                    "table_name": "orders_table",
                    "index_name": "idx_orders_id",
                    "usage_metrics": {"usage_ratio": 0.02},
                    "bloat_analysis": {"bloat_severity": "low", "estimated_bloat_percent": 15},
                    "risk_assessment": {"current_risk_level": "medium", "primary_risks": []},
                }
            ],
        },
        "locking_analysis": {
            "blocking_detected": False,
            "wait_event_type": None,
            "reasoning": None,
        },
        "incident_metadata": {
            "incident_id": "INC-20260125-8F2A1B",
            "incident_type": "QUERY_PERFORMANCE",
            "severity": "P2",
            "status": "ongoing",
        },
    }


def create_pgbench_data() -> Dict[str, Any]:
    """Create sample pgbench data."""
    return {
        "tps": 45,  # Below PG_LOW_TPS (100)
        "latency_ms": {"avg_ms": 350, "p95_ms": 500},
        "errors": 12,
    }


# ---------------------------------------------------------------------
# Integration Demonstration
# ---------------------------------------------------------------------

def demonstrate_integration():
    """Demonstrate RootCauseEngine integration with existing SignalEngine."""
    
    print("=" * 70)
    print("RootCauseEngine Integration Demonstration")
    print("=" * 70)
    
    # -----------------------------------------------------------------
    # Step 1: SignalEngine Processing (existing code)
    # -----------------------------------------------------------------
    print("\n[STEP 1] SignalEngine Processing")
    print("-" * 40)
    
    # Create test data
    incident_data = create_slow_query_incident_data()
    
    # Process with SignalEngine (existing)
    signal_engine = SignalEngine()
    signal_result = signal_engine.process(incident_data)
    
    print(f"  Signals generated: {len(signal_result.signals)}")
    print(f"  Analysis: {signal_result.analysis}")
    print(f"  Processing time: {signal_result.processing_time:.4f}s")
    
    # Print each signal
    print("\n  Generated Signals:")
    for i, signal in enumerate(signal_result.signals, 1):
        print(f"    {i}. [{signal.severity.upper()}] {signal.name}")
        print(f"       Type: {signal.type}, Confidence: {signal.confidence}")
    
    # -----------------------------------------------------------------
    # Step 2: RootCauseEngine Analysis (new post-processor)
    # -----------------------------------------------------------------
    print("\n[STEP 2] RootCauseEngine Analysis")
    print("-" * 40)
    
    # Create and run RootCauseEngine
    engine = RootCauseEngine(log_level="INFO")
    results = engine.analyze(signal_result)
    
    print(f"  Categories analyzed: {len(results)}")
    
    likely_causes = [
        (cat, res) for cat, res in results.items()
        if res.is_likely_cause
    ]
    print(f"  Likely causes identified: {len(likely_causes)}")
    
    # -----------------------------------------------------------------
    # Step 3: Detailed Results
    # -----------------------------------------------------------------
    print("\n[STEP 3] Detailed Root Cause Results")
    print("-" * 40)
    
    primary_causes = engine.get_primary_causes(results, top_n=5)
    
    for i, (category, result) in enumerate(primary_causes, 1):
        print(f"\n  {i}. {category.value}")
        print(f"     Confidence: {result.confidence:.0%}")
        print(f"     Evidence IDs: {result.evidence_ids}")
        print(f"     Matched Rules: {len(result.matched_rules)}")
        
        if result.contributing_factors:
            print(f"     Contributing Factors:")
            for factor in result.contributing_factors[:3]:
                print(f"       • {factor}")
        
        if result.recommendations:
            print(f"     Recommendations:")
            for rec in result.recommendations[:2]:
                print(f"       → {rec}")
        
        if result.false_positive_notes:
            print(f"     False Positive Notes:")
            for fp in result.false_positive_notes[:1]:
                print(f"       ⚠ {fp}")
    
    # -----------------------------------------------------------------
    # Step 4: Formatted Output
    # -----------------------------------------------------------------
    print("\n[STEP 4] Formatted Output")
    print("-" * 40)
    
    formatted = engine.format_results(results)
    
    print(f"  Analysis Timestamp: {formatted['analysis_timestamp']}")
    print(f"  Categories Analyzed: {formatted['categories_analyzed']}")
    print(f"  Likely Causes: {formatted['likely_causes_count']}")
    
    print("\n  Primary Causes Summary:")
    for cause in formatted['primary_causes']:
        print(f"    • {cause['category']}: {cause['confidence']:.0%} confidence")
    
    # -----------------------------------------------------------------
    # Step 5: Full JSON Output
    # -----------------------------------------------------------------
    print("\n[STEP 5] Full JSON Output")
    print("-" * 40)
    
    # Convert results to serializable format
    json_output = {
        "analysis_timestamp": datetime.utcnow().isoformat(),
        "signal_count": len(signal_result.signals),
        "results": {
            cat.value: res.to_dict()
            for cat, res in sorted(
                results.items(),
                key=lambda x: x[1].confidence,
                reverse=True
            )
        },
    }
    
    print(json.dumps(json_output, indent=2, default=str)[:1000] + "...")
    
    return results


def demonstrate_false_positive_handling():
    """Demonstrate false positive detection and handling."""
    
    print("\n" + "=" * 70)
    print("False Positive Handling Demonstration")
    print("=" * 70)
    
    engine = RootCauseEngine()
    
    # Create a mock signal result with small table (should trigger FP)
    small_table_signals = [
        Signal(
            id="sig_small_001",
            name="missing_index_large_table",
            type="json_context",
            severity="high",
            confidence=0.90,
            data={"row_count": 5000},  # Small table - false positive
            metadata={"metric_name": "row_count", "metric_value": 5000},
        ),
    ]
    
    small_table_result = SignalResult(
        signals=small_table_signals,
        analysis={"signal_count": 1},
        filtered_count=0,
        processing_time=0.001,
    )
    
    print("\n  Test Case: Small Table (5000 rows)")
    print("  Expected: False positive for 'small_table'")
    
    results = engine.analyze(small_table_result)
    
    for category, result in results.items():
        if result.is_likely_cause:
            print(f"\n  Category: {category.value}")
            print(f"  Confidence: {result.confidence:.0%}")
            print(f"  False Positive Notes:")
            for fp in result.false_positive_notes:
                print(f"    ⚠ {fp}")
    
    # Compare with large table
    large_table_signals = [
        Signal(
            id="sig_large_001",
            name="missing_index_large_table",
            type="json_context",
            severity="high",
            confidence=0.90,
            data={"row_count": 10000000},  # Large table - no FP
            metadata={"metric_name": "row_count", "metric_value": 10000000},
        ),
    ]
    
    large_table_result = SignalResult(
        signals=large_table_signals,
        analysis={"signal_count": 1},
        filtered_count=0,
        processing_time=0.001,
    )
    
    print("\n  Test Case: Large Table (10M rows)")
    print("  Expected: No false positives")
    
    results = engine.analyze(large_table_result)
    
    for category, result in results.items():
        if result.is_likely_cause:
            print(f"\n  Category: {category.value}")
            print(f"  Confidence: {result.confidence:.0%}")
            print(f"  False Positive Notes: {len(result.false_positive_notes)}")


def demonstrate_multi_cause_scenario():
    """Demonstrate detection of multiple root causes."""
    
    print("\n" + "=" * 70)
    print("Multi-Cause Detection Demonstration")
    print("=" * 70)
    
    engine = RootCauseEngine()
    
    # Create signals that trigger multiple root causes
    multi_cause_signals = [
        # Index issues
        Signal(
            id="sig_001",
            name="sequential_scan_detected",
            type="query_metrics",
            severity="high",
            confidence=0.90,
            data={"seq_scan_ratio": 0.99},
            metadata={"metric_name": "seq_scan_ratio", "metric_value": 0.99},
        ),
        Signal(
            id="sig_002",
            name="missing_index_large_table",
            type="json_context",
            severity="high",
            confidence=0.90,
            data={"row_count": 10000000},
            metadata={"metric_name": "row_count", "metric_value": 10000000},
        ),
        # Statistics issues
        Signal(
            id="sig_003",
            name="stale_statistics",
            type="table_stats",
            severity="medium",
            confidence=0.85,
            data={"days_since_analyze": 7},
            metadata={"metric_name": "days_since_analyze", "metric_value": 7},
        ),
        # Locking issues
        Signal(
            id="sig_004",
            name="blocking_detected",
            type="incident",
            severity="high",
            confidence=0.95,
            data={"wait_event": "Lock"},
        ),
    ]
    
    multi_cause_result = SignalResult(
        signals=multi_cause_signals,
        analysis={"signal_count": 4},
        filtered_count=0,
        processing_time=0.001,
    )
    
    print(f"\n  Input Signals: {len(multi_cause_signals)}")
    
    results = engine.analyze(multi_cause_result)
    
    likely_causes = [
        (cat, res) for cat, res in results.items()
        if res.is_likely_cause
    ]
    
    print(f"  Likely Causes Detected: {len(likely_causes)}")
    
    for category, result in likely_causes:
        print(f"\n  {category.value}:")
        print(f"    Confidence: {result.confidence:.0%}")
        print(f"    Evidence: {len(result.evidence_ids)} items")
        print(f"    Rules: {len(result.matched_rules)} matched")
        print(f"    Recommendations: {len(result.recommendations)}")


def demonstrate_pgbench_integration():
    """Demonstrate integration with pgbench data."""
    
    print("\n" + "=" * 70)
    print("pgbench Data Integration Demonstration")
    print("=" * 70)
    
    signal_engine = SignalEngine()
    engine = RootCauseEngine()
    
    # Process pgbench data
    pgbench_data = create_pgbench_data()
    signal_result = signal_engine.process(pgbench_data)
    
    print(f"\n  pgbench Data: TPS={pgbench_data['tps']}, "
          f"Latency={pgbench_data['latency_ms']['avg_ms']}ms")
    print(f"  Signals Generated: {len(signal_result.signals)}")
    
    if signal_result.signals:
        results = engine.analyze(signal_result)
        
        print(f"\n  Root Cause Analysis:")
        for category, result in results.items():
            if result.is_likely_cause:
                print(f"    • {category.value}: {result.confidence:.0%}")
        
        # Show recommendations
        for category, result in results.items():
            if result.is_likely_cause and result.recommendations:
                print(f"\n    Recommendations for {category.value}:")
                for rec in result.recommendations[:2]:
                    print(f"      → {rec}")
    else:
        print("  No signals generated (TPS above threshold)")


def show_rule_coverage():
    """Show the rule coverage summary."""
    
    print("\n" + "=" * 70)
    print("Rule Coverage Summary")
    print("=" * 70)
    
    engine = RootCauseEngine()
    
    print(f"\n  Total Rules: {len(engine.registry.rules)}")
    print(f"  Total Categories: {len(engine.registry.get_all_categories())}")
    
    print("\n  Rules by Category:")
    for category in RootCauseCategory:
        rules = engine.registry.get_rules_for_category(category)
        print(f"\n    {category.value} ({len(rules)} rules):")
        for rule in rules:
            fp_count = len(rule.false_positives)
            print(f"      • {rule.name}")
            print(f"        Confidence: {rule.confidence_contribution:.0%} → "
                  f"max {rule.max_confidence:.0%}")
            if fp_count:
                print(f"        False Positives: {fp_count}")


def main():
    """Run all demonstrations."""
    
    # Run demonstrations
    demonstrate_integration()
    demonstrate_false_positive_handling()
    demonstrate_multi_cause_scenario()
    demonstrate_pgbench_integration()
    show_rule_coverage()
    
    print("\n" + "=" * 70)
    print("Integration Demonstration Complete")
    print("=" * 70)
    print("""
Key Takeaways:
1. RootCauseEngine integrates as a post-processor to SignalEngine
2. It identifies multiple root causes with confidence scores
3. Each rule tracks false positives and provides recommendations
4. No modification to existing SignalEngine code required
5. Deterministic and auditable - no LLM involved
""")


if __name__ == "__main__":
    main()

