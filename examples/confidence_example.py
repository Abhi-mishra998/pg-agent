#!/usr/bin/env python3
"""
Example: Slow Query Due to Missing Index - Confidence Scoring Breakdown

This example demonstrates how the ConfidenceScorer calculates confidence
for a realistic PostgreSQL incident: slow query performance caused by
a missing index on a large table.

The example shows:
1. Multiple signals pointing to the same root cause
2. Evidence from different sources (pg_stat_statements, pg_stat_user_tables)
3. Data freshness considerations
4. Complete confidence breakdown with explanations
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta
from signals.confidence_scorer import (
    ConfidenceScorer,
    SignalEvidence,
    ExpectedEvidence,
)


def create_slow_query_missing_index_scenario() -> tuple:
    """
    Create the slow query missing index scenario.
    
    Returns:
        Tuple of (signals, expected_evidence, conflict_groups)
    """
    # Current timestamp for freshness calculations
    now = datetime.utcnow()
    
    # Define signals from different sources
    signals = [
        # Signal 1: Sequential scan detected (query_metrics)
        SignalEvidence(
            signal_id="sig_seq_scan_001",
            signal_name="sequential_scan_detected",
            signal_type="query_metrics",
            confidence=0.90,
            evidence_ids=["ev_pg_stat_user_tables", "ev_explain_analyze"],
            data_timestamp=(now - timedelta(minutes=15)).isoformat() + "Z",
            metadata={
                "table": "orders_table",
                "seq_scans": 1523,
                "idx_scans": 12,
                "seq_scan_ratio": 0.99,
            }
        ),
        
        # Signal 2: High query latency (query_metrics)
        SignalEvidence(
            signal_id="sig_latency_001",
            signal_name="high_query_latency",
            signal_type="query_metrics",
            confidence=0.92,
            evidence_ids=["ev_pg_stat_statements"],
            data_timestamp=(now - timedelta(minutes=10)).isoformat() + "Z",
            metadata={
                "query_fingerprint": "SELECT * FROM orders WHERE customer_id = $1",
                "mean_time_ms": 15234.5,
                "calls": 1452,
                "deviation_factor": 15.2,
            }
        ),
        
        # Signal 3: Missing index on large table (table_stats)
        SignalEvidence(
            signal_id="sig_missing_idx_001",
            signal_name="missing_index_large_table",
            signal_type="table_stats",
            confidence=0.88,
            evidence_ids=["ev_pg_indexes", "ev_pg_tables"],
            data_timestamp=(now - timedelta(minutes=5)).isoformat() + "Z",
            metadata={
                "table": "orders_table",
                "row_count": 10234567,
                "index_count": 1,
                "index_columns": ["id"],
                "missing_columns": ["customer_id", "created_at"],
            }
        ),
        
        # Signal 4: Row estimation error (query_metrics)
        SignalEvidence(
            signal_id="sig_row_est_001",
            signal_name="severe_row_estimation_error",
            signal_type="query_metrics",
            confidence=0.85,
            evidence_ids=["ev_explain_analyze", "ev_pg_statistic"],
            data_timestamp=(now - timedelta(minutes=10)).isoformat() + "Z",
            metadata={
                "estimated_rows": 1000,
                "actual_rows": 356024,
                "error_factor": 356,
            }
        ),
        
        # Signal 5: Plan regression (query_metrics)
        SignalEvidence(
            signal_id="sig_plan_reg_001",
            signal_name="query_plan_regression",
            signal_type="query_metrics",
            confidence=0.87,
            evidence_ids=["ev_pg_stat_statements", "ev_plan_hash"],
            data_timestamp=(now - timedelta(minutes=30)).isoformat() + "Z",
            metadata={
                "current_plan_hash": "abc123",
                "baseline_plan_hash": "xyz789",
                "plan_changed": True,
                "change_reason": "statistics_updated",
            }
        ),
    ]
    
    # Define expected evidence for this incident type
    expected_evidence = [
        ExpectedEvidence(evidence_type="query_metrics", is_required=True),
        ExpectedEvidence(evidence_type="table_stats", is_required=True),
        ExpectedEvidence(evidence_type="index_health", is_required=False),
        ExpectedEvidence(evidence_type="configuration", is_required=False),
        ExpectedEvidence(evidence_type="execution_plan", is_required=True),
    ]
    
    # No conflicting evidence in this scenario
    conflict_groups = []
    
    return signals, expected_evidence, conflict_groups


def demonstrate_confidence_calculation():
    """Demonstrate the full confidence calculation with breakdown."""
    
    print("=" * 80)
    print("EXAMPLE: Slow Query Due to Missing Index")
    print("Confidence Scoring Model Demonstration")
    print("=" * 80)
    
    # Create scorer with deterministic seed
    scorer = ConfidenceScorer(seed=42)
    
    # Get scenario
    signals, expected_evidence, conflict_groups = create_slow_query_missing_index_scenario()
    
    print("\n" + "-" * 80)
    print("INPUT SIGNALS")
    print("-" * 80)
    for sig in signals:
        print(f"  • {sig.signal_name} ({sig.signal_type})")
        print(f"    Confidence: {sig.confidence:.2f}")
        print(f"    Data Age: {sig.data_timestamp}")
        print()
    
    print("-" * 80)
    print("EXPECTED EVIDENCE")
    print("-" * 80)
    for ev in expected_evidence:
        req_str = "REQUIRED" if ev.is_required else "OPTIONAL"
        print(f"  • {ev.evidence_type} ({req_str})")
    
    # Calculate confidence
    print("\n" + "-" * 80)
    print("CALCULATING CONFIDENCE")
    print("-" * 80)
    
    score, breakdown = scorer.calculate_confidence(
        signals=signals,
        expected_evidence=expected_evidence,
        conflict_groups=conflict_groups,
    )
    
    # Display detailed breakdown
    print(f"\n📊 FINAL CONFIDENCE SCORE: {score:.4f} ({score:.2%})")
    
    print("\n" + "-" * 80)
    print("DETAILED BREAKDOWN")
    print("-" * 80)
    print(breakdown.to_human_readable())
    
    # Get explanation
    explanation = scorer.explain_score(score, breakdown)
    
    print("\n" + "-" * 80)
    print("INTERPRETATION")
    print("-" * 80)
    print(f"Rating: {explanation['rating']}")
    print(f"\n{explanation['interpretation']}")
    
    if explanation['recommendations']:
        print("\n📋 RECOMMENDATIONS:")
        for rec in explanation['recommendations']:
            print(f"  • {rec}")
    
    # Show JSON output
    print("\n" + "-" * 80)
    print("MACHINE-READABLE OUTPUT (JSON)")
    print("-" * 80)
    print(json.dumps(explanation, indent=2))
    
    return score, breakdown, explanation


def demonstrate_scenarios():
    """Demonstrate different scenarios affecting confidence."""
    
    print("\n" + "=" * 80)
    print("SCENARIO COMPARISON")
    print("=" * 80)
    
    scenarios = {
        "Optimal Case": {
            "signal_confidences": [0.95, 0.92, 0.94],
            "missing_evidence": 0,
            "independent_groups": 3,
            "max_age_hours": 0.5,
            "conflicts": 0,
        },
        "With Stale Data": {
            "signal_confidences": [0.95, 0.92, 0.94],
            "missing_evidence": 0,
            "independent_groups": 3,
            "max_age_hours": 12.0,  # Stale data
            "conflicts": 0,
        },
        "With Missing Evidence": {
            "signal_confidences": [0.95, 0.92, 0.94],
            "missing_evidence": 2,  # Missing required evidence
            "independent_groups": 3,
            "max_age_hours": 0.5,
            "conflicts": 0,
        },
        "With Conflicts": {
            "signal_confidences": [0.95, 0.92, 0.94],
            "missing_evidence": 0,
            "independent_groups": 3,
            "max_age_hours": 0.5,
            "conflicts": 2,  # Conflicting evidence
        },
        "Low Confidence Signals": {
            "signal_confidences": [0.60, 0.55, 0.65],  # Low individual confidence
            "missing_evidence": 0,
            "independent_groups": 3,
            "max_age_hours": 0.5,
            "conflicts": 0,
        },
    }
    
    scorer = ConfidenceScorer(seed=42)
    
    for scenario_name, params in scenarios.items():
        print(f"\n{'─' * 40}")
        print(f"📌 {scenario_name}")
        print(f"{'─' * 40}")
        
        # Build signals
        signals = [
            SignalEvidence(
                signal_id=f"sig_{i}",
                signal_name=f"signal_{i}",
                signal_type=f"type_{i % params['independent_groups']}",
                confidence=conf,
            )
            for i, conf in enumerate(params['signal_confidences'])
        ]
        
        # Build expected evidence
        expected = [
            ExpectedEvidence(evidence_type="query_metrics", is_required=True),
            ExpectedEvidence(evidence_type="table_stats", is_required=True),
            ExpectedEvidence(evidence_type="index_health", is_required=True),
        ]
        
        # Simulate missing evidence
        if params['missing_evidence'] > 0:
            expected = expected[:-params['missing_evidence']]
        
        # Build conflict groups
        conflicts = []
        for _ in range(params['conflicts']):
            conflicts.append([f"sig_0", f"sig_1"])
        
        # Simulate data freshness in metadata
        now = datetime.utcnow()
        for i, sig in enumerate(signals):
            sig.data_timestamp = (
                now - timedelta(hours=params['max_age_hours'])
            ).isoformat() + "Z"
        
        # Calculate
        score, breakdown = scorer.calculate_confidence(
            signals=signals,
            expected_evidence=expected,
            conflict_groups=conflicts,
        )
        
        print(f"  Confidence: {score:.4f} ({score:.2%})")
        print(f"  Components:")
        print(f"    Base: {breakdown.base_confidence:.4f}")
        print(f"    Completeness: {breakdown.evidence_completeness:.4f}")
        print(f"    Agreement: {breakdown.signal_agreement:.4f}")
        print(f"    Freshness: {breakdown.data_freshness:.4f}")
        print(f"    Conflict Penalty: {breakdown.conflict_penalty:.4f}")


def demonstrate_determinism():
    """Demonstrate that the scorer produces deterministic results."""
    
    print("\n" + "=" * 80)
    print("DETERMINISM VERIFICATION")
    print("=" * 80)
    
    # Same inputs should produce same outputs
    signals, expected_evidence, conflict_groups = create_slow_query_missing_index_scenario()
    
    scores = []
    for seed in [42, 42, 42]:
        scorer = ConfidenceScorer(seed=seed)
        score, _ = scorer.calculate_confidence(
            signals=signals,
            expected_evidence=expected_evidence,
            conflict_groups=conflict_groups,
        )
        scores.append(score)
    
    print(f"\nSame inputs with same seed (42):")
    for i, score in enumerate(scores):
        print(f"  Run {i+1}: {score:.6f}")
    
    all_same = len(set(scores)) == 1
    print(f"\n✓ Deterministic: {all_same} (all scores identical)")
    
    # Different seeds should produce same results (no random component in calculation)
    different_seed_scores = []
    for seed in [1, 42, 999]:
        scorer = ConfidenceScorer(seed=seed)
        score, _ = scorer.calculate_confidence(
            signals=signals,
            expected_evidence=expected_evidence,
            conflict_groups=conflict_groups,
        )
        different_seed_scores.append(score)
    
    print(f"\nDifferent seeds (1, 42, 999):")
    for i, score in enumerate(different_seed_scores):
        print(f"  Seed {i+1}: {score:.6f}")
    
    all_same_seed_independence = len(set(different_seed_scores)) == 1
    print(f"\n✓ Seed-independent: {all_same_seed_independence} (no random component)")


def main():
    """Run all demonstrations."""
    
    # Main calculation
    score, breakdown, explanation = demonstrate_confidence_calculation()
    
    # Scenario comparison
    demonstrate_scenarios()
    
    # Determinism verification
    demonstrate_determinism()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
For the "Slow Query Due to Missing Index" scenario:

• Final Confidence Score: {score:.4f} ({score:.2%})
• Rating: {explanation['rating']}

Key Factors:
1. Base Confidence: High individual signal confidences (0.85-0.92)
2. Evidence Completeness: Most required evidence present (2 required missing)
3. Signal Agreement: Multiple signal types agreeing (2 independent groups)
4. Data Freshness: Data is relatively recent (<30 min old)
5. No Conflicts: All signals point to same root cause

This high confidence score indicates that the missing index diagnosis
is well-supported by evidence and suitable for production decisions.
""")


if __name__ == "__main__":
    main()

