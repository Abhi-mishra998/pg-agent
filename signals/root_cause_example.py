#!/usr/bin/env python3
"""
Example: RootCauseEngine Integration Demo

This demonstrates how RootCauseEngine integrates with existing SignalEngine
as a post-processor WITHOUT modifying SignalEngine.

Pipeline:
  SignalEngine.process() → SignalResult → RootCauseEngine.analyze() → RootCauseResults
                                                        ↓
                                              IncidentOutputFormatter
                                                        ↓
                                              IncidentOutput (final)
"""

from signals.signal_engine import SignalEngine, Signal
from signals.root_cause_engine import RootCauseEngine, RootCauseCategory
from signals.incident_schema import IncidentOutputFormatter
from datetime import datetime


def create_sample_signal_result() -> dict:
    """Create sample signals representing a slow query incident."""
    now = int(datetime.utcnow().timestamp() * 1000)
    
    # Simulate signals that would come from SignalEngine
    return {
        "signals": [
            Signal(
                id=f"sig_seq_scan_{now}",
                name="sequential_scan_detected",
                type="query_metrics",
                severity="high",
                confidence=0.88,
                data={
                    "table": "orders_table",
                    "seq_scans": 1523,
                    "idx_scans": 12,
                    "seq_scan_ratio": 0.99,
                },
                metadata={
                    "explain": "Sequential scan on large table (10M rows)",
                    "metric_name": "seq_scan_ratio",
                    "metric_value": 0.99,
                },
            ),
            Signal(
                id=f"sig_latency_{now}",
                name="high_query_latency",
                type="query_metrics",
                severity="high",
                confidence=0.92,
                data={
                    "mean_time_ms": 15234.5,
                    "max_time_ms": 45678.9,
                    "calls": 1452,
                },
                metadata={
                    "explain": "Query execution time 30x higher than baseline",
                    "metric_name": "mean_time_ms",
                    "metric_value": 15234.5,
                },
            ),
            Signal(
                id=f"sig_missing_idx_{now}",
                name="missing_index_large_table",
                type="json_context",
                severity="high",
                confidence=0.90,
                data={
                    "table": "orders_table",
                    "row_count": 10234567,
                    "missing_columns": ["customer_id", "created_at"],
                },
                metadata={
                    "explain": "Large table lacks index on filter columns",
                },
            ),
            Signal(
                id=f"sig_stale_stats_{now}",
                name="stale_statistics",
                type="table_stats",
                severity="medium",
                confidence=0.75,
                data={
                    "last_analyze": "2026-01-18T03:00:00Z",
                    "days_since_analyze": 7,
                },
                metadata={
                    "explain": "Table statistics are 7 days old",
                },
            ),
        ],
        "analysis": {
            "signal_count": 4,
            "severities": {"high": 3, "medium": 1},
            "highest_severity": "high",
        },
        "filtered_count": 0,
        "processing_time": 0.045,
    }


def demonstrate_root_cause_engine():
    """Demonstrate the RootCauseEngine analysis."""
    print("=" * 70)
    print("RootCauseEngine Integration Demo")
    print("=" * 70)
    
    # Step 1: SignalEngine produces signals (existing code)
    print("\n[1] SignalEngine Processing (existing)")
    signal_data = create_sample_signal_result()
    signal_result = type('SignalResult', (), signal_data)()
    print(f"    - Generated {len(signal_result.signals)} signals")
    for sig in signal_result.signals:
        print(f"      • {sig.name} (severity={sig.severity}, confidence={sig.confidence})")
    
    # Step 2: RootCauseEngine analyzes signals (NEW - post-processor)
    print("\n[2] RootCauseEngine Analysis (new post-processor)")
    engine = RootCauseEngine(log_level="INFO")
    results = engine.analyze(signal_result)
    
    print(f"    - Analyzed {len(results)} root cause categories")
    print(f"    - Found {len([r for r in results.values() if r.is_likely_cause])} likely causes")
    
    # Display results
    primary_causes = engine.get_primary_causes(results, top_n=3)
    print("\n    Primary Root Causes (sorted by confidence):")
    for i, (category, result) in enumerate(primary_causes, 1):
        print(f"\n    {i}. {category.value} (confidence: {result.confidence})")
        print(f"       Evidence IDs: {result.evidence_ids}")
        print(f"       Contributing factors:")
        for factor in result.contributing_factors[:2]:
            print(f"         • {factor}")
        print(f"       Recommendations:")
        for rec in result.recommendations[:2]:
            print(f"         • {rec}")
        if result.false_positive_notes:
            print(f"       False positive notes:")
            for fp in result.false_positive_notes[:1]:
                print(f"         ⚠ {fp}")
    
    # Step 3: Full formatted output
    print("\n[3] Formatted Results")
    formatted = engine.format_results(results)
    print(f"    Primary causes identified: {formatted['likely_causes_count']}")
    for cause in formatted['primary_causes']:
        print(f"    • {cause['category']}: {cause['confidence']} confidence")
    
    # Step 4: Demonstrate IncidentOutput integration
    print("\n[4] Integration with IncidentOutput (optional)")
    print("    RootCauseEngine results can feed into IncidentOutputFormatter")
    print("    to produce the final incident report schema.")
    
    return results


def show_rule_coverage():
    """Show which rules exist for each category."""
    engine = RootCauseEngine()
    
    print("\n" + "=" * 70)
    print("Rule Registry Overview")
    print("=" * 70)
    
    categories = engine.registry.get_all_categories()
    print(f"\nTotal categories: {len(categories)}")
    print(f"Total rules: {len(engine.registry.rules)}")
    
    print("\nRules per category:")
    for category in RootCauseCategory:
        rules = engine.registry.get_rules_for_category(category)
        print(f"\n  {category.value} ({len(rules)} rules):")
        for rule in rules:
            fp_count = len(rule.false_positives)
            print(f"    • {rule.name}")
            print(f"      Required: {rule.required_evidence}")
            print(f"      Confidence: {rule.confidence_contribution} (max: {rule.max_confidence})")
            if fp_count > 0:
                print(f"      False positives: {fp_count} scenarios tracked")


def show_false_positive_tracking():
    """Demonstrate false positive detection."""
    print("\n" + "=" * 70)
    print("False Positive Tracking Example")
    print("=" * 70)
    
    engine = RootCauseEngine()
    
    # Find the sequential scan rule
    rule = None
    for r in engine.registry.rules:
        if r.name == "sequential_scan_large_table":
            rule = r
            break
    
    if rule:
        print(f"\nRule: {rule.name}")
        print(f"Category: {rule.category.value}")
        print(f"Required evidence: {rule.required_evidence}")
        print(f"False positive scenarios:")
        for fp in rule.false_positives:
            print(f"  • {fp}")
        
        print("\nWhen matching against signals:")
        print("  - If table has < 10000 rows → false positive flagged")
        print("  - Confidence is reduced by 10% per false positive")
        print("  - Warning is added to result")


if __name__ == "__main__":
    # Run demo
    demonstrate_root_cause_engine()
    
    # Show rule overview
    show_rule_coverage()
    
    # Show false positive tracking
    show_false_positive_tracking()
    
    print("\n" + "=" * 70)
    print("Integration Summary")
    print("=" * 70)
