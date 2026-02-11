#!/usr/bin/env python3
"""
Evidence Explorer Example

Demonstrates the Evidence Explorer capabilities with sample data.
Run this script to see the Evidence Explorer in action.

Usage:
    cd /Users/abhishekmishra/pg-agent
    python3 examples/evidence_explorer_example.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from signals.evidence_explorer import EvidenceExplorer, create_evidence_explorer
from signals.evidence_types import EvidenceSource, EvidenceType
from signals.evidence_card import EvidenceCard, EvidenceProvenance
from signals.confidence_scorer import ConfidenceScorer


def create_sample_evidence() -> EvidenceExplorer:
    """Create an EvidenceExplorer with sample data from the slow query incident."""
    
    explorer = create_evidence_explorer(
        incident_id="INC-20260125-8F2A1B",
    )
    
    # =========================================================================
    # METRIC EVIDENCE - From pg_stat_statements
    # =========================================================================
    
    # High query latency
    explorer.add_pg_stat_statements_evidence(
        query_hash="abc123def456",
        mean_time_ms=15234.5,
        calls=1452,
        query_text="SELECT * FROM orders_table WHERE customer_id = $1 AND created_at > $2",
        baseline_ms=120.0,
    )
    
    # Sequential scan detected
    explorer.add_table_stats_evidence(
        table_name="orders_table",
        seq_scans=1523,
        idx_scans=12,
        row_count=10234567,
        last_analyze_days=7,
    )
    
    # =========================================================================
    # CONFIG EVIDENCE - From pg_settings
    # =========================================================================
    
    explorer.add_config_evidence(
        param_name="work_mem",
        current_value=4,  # 4 MB
        recommended_value=64,  # 64 MB
        unit="MB",
        context="Hash join requires 256MB",
    )
    
    # =========================================================================
    # LOCK EVIDENCE - From pg_locks
    # =========================================================================
    
    explorer.add_lock_evidence(
        blocked_pid=34121,
        blocking_pid=33218,
        wait_duration_seconds=180,
        lock_type="RowExclusiveLock",
        blocked_query="SELECT * FROM orders WHERE customer_id = $1",
        blocking_query="UPDATE orders SET status = 'CLOSED' WHERE updated_at < $1",
    )
    
    # =========================================================================
    # FACT EVIDENCE - Table structure
    # =========================================================================
    
    fact_card = EvidenceCard(
        evidence_id="ev_fact_001",
        evidence_type=EvidenceType.FACT,
        title="Table has 10.2M rows, 1 index",
        description="Large table orders_table has only index on (id) column",
        confidence=0.95,
        provenance=EvidenceProvenance.from_source(
            EvidenceSource.PG_INDEXES,
            datetime.utcnow().isoformat()
        ),
        context={
            "table": "orders_table",
            "row_count": 10234567,
            "indexes": ["orders_table_pkey"],
            "missing_columns": ["customer_id", "created_at"],
        },
        raw_value={
            "table": "orders_table",
            "indexes": [{"name": "orders_table_pkey", "columns": ["id"]}],
        }
    )
    explorer.add_evidence(fact_card)
    
    # =========================================================================
    # MISSING EVIDENCE - Data gaps
    # =========================================================================
    
    explorer.add_missing_evidence(
        evidence_type="pg_statio_user_tables",
        description="Cache hit ratio not available - pg_statio not enabled",
        required_for=["MISSING_INDEX", "STALE_STATISTICS"],
        confidence_penalty=0.10,
        collection_action="CREATE EXTENSION IF NOT EXISTS pg_stat_statements;",
        collection_risk="low",
    )
    
    return explorer


def demonstrate_evidence_explorer():
    """Demonstrate all Evidence Explorer capabilities."""
    
    print("=" * 80)
    print("EVIDENCE EXPLORER DEMONSTRATION")
    print("Goal: Make a senior DBA say 'This system is not guessing.'")
    print("=" * 80)
    print()
    
    # Create explorer with sample data
    explorer = create_sample_evidence()
    
    # =========================================================================
    # 1. Terminal Output
    # =========================================================================
    
    print("1. TERMINAL OUTPUT")
    print("-" * 80)
    print()
    terminal_output = explorer.render_terminal(include_details=True)
    print(terminal_output)
    print()
    
    # =========================================================================
    # 2. Markdown Output
    # =========================================================================
    
    print("2. MARKDOWN OUTPUT (snippet)")
    print("-" * 80)
    print()
    markdown_output = explorer.render_markdown(include_details=True)
    # Show first 100 lines
    lines = markdown_output.split('\n')[:100]
    print('\n'.join(lines))
    print()
    print(f"... [truncated, full output has {len(markdown_output.split(chr(10)))} lines]")
    print()
    
    # =========================================================================
    # 3. Confidence Breakdown
    # =========================================================================
    
    print("3. CONFIDENCE BREAKDOWN")
    print("-" * 80)
    print()
    breakdown = explorer.calculate_confidence_breakdown()
    print(f"Overall Confidence: {breakdown['overall_confidence']:.0%}")
    print()
    print("Components:")
    for key, value in breakdown.items():
        if key != "overall_confidence":
            print(f"  - {key.replace('_', ' ').title()}: {value:.0%}")
    print()
    
    # =========================================================================
    # 4. Source Summary
    # =========================================================================
    
    print("4. EVIDENCE SOURCE SUMMARY")
    print("-" * 80)
    print()
    source_counts = explorer.get_source_summary()
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count} evidence items")
    print()
    
    # =========================================================================
    # 5. Evidence Groups
    # =========================================================================
    
    print("5. EVIDENCE GROUPED BY TYPE")
    print("-" * 80)
    print()
    groups = explorer.group_evidence()
    for group in groups:
        print(f"  {group.icon} {group.display_name}: {group.count} items")
        for card in group.evidence:
            print(f"    - {card.title} (confidence: {card.confidence:.0%})")
    print()
    
    # =========================================================================
    # 6. Contradiction Analysis
    # =========================================================================
    
    print("6. CONTRADICTION ANALYSIS")
    print("-" * 80)
    print()
    contradictions = explorer.analyze_contradictions()
    if contradictions:
        for contra in contradictions:
            print(f"  Claim: {contra.claim}")
            print(f"  Status: {contra.resolution_status}")
            if contra.explanation:
                print(f"  Explanation: {contra.explanation}")
    else:
        print("  No contradictions detected in this evidence set.")
    print()
    
    # =========================================================================
    # 7. Missing Evidence
    # =========================================================================
    
    print("7. MISSING EVIDENCE")
    print("-" * 80)
    print()
    for missing in explorer.collection.missing_evidence:
        print(f"  Type: {missing.evidence_type}")
        print(f"  Description: {missing.description}")
        print(f"  Confidence Penalty: -{missing.confidence_penalty:.0%}")
        if missing.collection_action:
            print(f"  Collection Action: {missing.collection_action}")
        print()
    
    # =========================================================================
    # 8. JSON Export
    # =========================================================================
    
    print("8. JSON EXPORT (snippet)")
    print("-" * 80)
    print()
    data = explorer.to_dict()
    json_str = json.dumps(data, indent=2, default=str)
    # Show first 50 lines
    lines = json_str.split('\n')[:50]
    print('\n'.join(lines))
    print()
    print(f"... [truncated, full JSON has {len(json_str.split(chr(10)))} lines]")
    print()
    
    # =========================================================================
    # 9. Save to File
    # =========================================================================
    
    print("9. SAVE TO FILE")
    print("-" * 80)
    print()
    output_path = "/Users/abhishekmishra/pg-agent/data/output/evidence_explorer_output.json"
    explorer.save_to_file(output_path)
    print(f"Saved evidence collection to: {output_path}")
    print()
    
    # =========================================================================
    # 10. Load from File
    # =========================================================================
    
    print("10. LOAD FROM FILE")
    print("-" * 80)
    print()
    loaded_explorer = EvidenceExplorer.load_from_file(output_path, "INC-LOADED")
    print(f"Loaded evidence collection with {len(loaded_explorer.collection.evidence)} evidence items")
    print()


def demonstrate_evidence_card_details():
    """Demonstrate individual evidence card details."""
    
    print("=" * 80)
    print("EVIDENCE CARD DETAIL DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create a metric evidence card
    card = EvidenceCard.from_metric(
        metric_name="seq_scan_count",
        current_value=1523,
        evidence_type=EvidenceType.TABLE_STATS,
        source=EvidenceSource.PG_STAT_USER_TABLES,
        baseline_value=50,
        unit="scans",
        description="Sequential scan detected on orders_table with 10.2M rows",
        confidence=0.95,
    )
    
    print("Evidence Card Fields:")
    print("-" * 80)
    print(f"  Evidence ID: {card.evidence_id}")
    print(f"  Evidence Type: {card.evidence_type.value}")
    print(f"  Title: {card.title}")
    print(f"  Description: {card.description}")
    print()
    
    print("Metric:")
    print("-" * 80)
    if card.metric:
        print(f"  Name: {card.metric.name}")
        print(f"  Value: {card.metric.value}")
        print(f"  Unit: {card.metric.unit}")
        print(f"  Threshold: {card.metric.threshold}")
        print(f"  Is Anomaly: {card.metric.is_anomaly}")
    print()
    
    print("Deviation:")
    print("-" * 80)
    if card.deviation:
        dev = card.get_deviation_display()
        print(f"  Current: {dev['current']}")
        print(f"  Baseline: {dev['baseline']}")
        print(f"  Percent: {dev['percent']}")
        print(f"  Severity: {dev['severity']}")
        print(f"  Icon: {dev['icon']}")
    print()
    
    print("Confidence Badge:")
    print("-" * 80)
    badge = card.get_confidence_badge()
    print(f"  Text: {badge['text']}")
    print(f"  Label: {badge['label']}")
    print(f"  Color: {badge['color']}")
    print(f"  Icon: {badge['icon']}")
    print()
    
    print("Provenance:")
    print("-" * 80)
    if card.provenance:
        print(f"  Source: {card.provenance.source_name}")
        print(f"  Icon: {card.provenance.icon}")
        print(f"  Color: {card.provenance.color}")
    print()


def demonstrate_contradiction_display():
    """Demonstrate how contradictions are displayed."""
    
    print("=" * 80)
    print("CONTRADICTION DISPLAY DEMONSTRATION")
    print("=" * 80)
    print()
    
    explorer = create_evidence_explorer("INC-CONTRA-001")
    
    # Manually add a contradiction
    from signals.evidence_card import Contradiction
    
    contradiction = Contradiction(
        claim="Query is using sequential scan",
        supporting_evidence=["ev_sig_001", "ev_sig_005"],
        contradicting_evidence=["ev_sig_006"],
        explanation="Index scan is on 'id' column, but query filters on 'customer_id'. "
                   "This confirms sequential scan for customer_id filter, not a contradiction.",
        resolution_status="resolved",
        uncertainty_impact=0.0,
    )
    explorer.collection.contradictions.append(contradiction)
    
    # Render contradiction display
    print("CONTRADICTION CARD:")
    print("-" * 80)
    print()
    print(f"⚠️  CLAIM: {contradiction.claim}")
    print()
    
    print("SUPPORTING EVIDENCE:")
    print("  🟢 CONFIRMED (2 sources)")
    for ev_id in contradiction.supporting_evidence:
        ev = explorer.collection.get_evidence_by_id(ev_id)
        if ev:
            print(f"    - {ev.title} (confidence: {ev.confidence:.0%})")
    print()
    
    print("CONFLICTING EVIDENCE:")
    print("  🔴 CONFLICT (1 source)")
    for ev_id in contradiction.contradicting_evidence:
        ev = explorer.collection.get_evidence_by_id(ev_id)
        if ev:
            print(f"    - {ev.title}")
    print()
    
    print("RESOLUTION:")
    print(f"  Status: {contradiction.resolution_status}")
    print(f"  Explanation: {contradiction.explanation}")
    print()
    
    # Unresolved contradiction example
    print("-" * 80)
    print("UNRESOLVED CONTRADICTION EXAMPLE:")
    print()
    
    unresolved = Contradiction(
        claim="Statistics are causing poor plan choice",
        supporting_evidence=["ev_sig_004"],
        contradicting_evidence=["ev_sig_008"],
        explanation="",
        resolution_status="unresolved",
        uncertainty_impact=0.08,
    )
    explorer.collection.contradictions.append(unresolved)
    
    print(f"⚠️  CLAIM: {unresolved.claim}")
    print()
    print("SUPPORTING:")
    print("  - days_since_analyze = 7 (threshold: 3)")
    print()
    print("CONFLICTING:")
    print("  - autovacuum_analyze_scale_factor = 0.01 (should trigger ANALYZE)")
    print()
    print("UNCERTAINTY IMPACT:")
    print(f"  Confidence Reduction: -{unresolved.uncertainty_impact:.0%}")
    print()
    print("QUESTIONS TO RESOLVE:")
    print("  1. Was there a bulk data load after last ANALYZE?")
    print("  2. Are distribution patterns changed?")
    print()


def demonstrate_missing_data_display():
    """Demonstrate how missing data is displayed."""
    
    print("=" * 80)
    print("MISSING DATA DISPLAY DEMONSTRATION")
    print("=" * 80)
    print()
    
    explorer = create_evidence_explorer("INC-MISSING-001")
    
    # Add missing evidence
    explorer.add_missing_evidence(
        evidence_type="pg_statio_user_tables",
        description="Cache hit ratio not available - pg_statio not enabled",
        required_for=["CACHE_PERFORMANCE"],
        confidence_penalty=0.10,
        collection_action="CREATE EXTENSION IF NOT EXISTS pg_stat_statements;",
        collection_risk="low",
        is_required=True,
    )
    
    explorer.add_missing_evidence(
        evidence_type="EXPLAIN ANALYZE",
        description="Actual execution plan not available",
        required_for=["BAD_PLAN"],
        confidence_penalty=0.25,
        collection_action="Run: EXPLAIN (ANALYZE, BUFFERS) <query>",
        collection_risk="low",
        is_required=True,
    )
    
    print("MISSING EVIDENCE PANEL:")
    print("-" * 80)
    print()
    print("To improve confidence, the following evidence is needed:")
    print()
    
    for missing in explorer.collection.missing_evidence:
        print(f"┌─ {missing.evidence_type}")
        print(f"│")
        print(f"│  Description: {missing.description}")
        print(f"│")
        print(f"│  Impact: -{missing.confidence_penalty:.0%} confidence penalty")
        print(f"│  Risk: {missing.collection_risk.upper()}")
        if missing.collection_action:
            print(f"│")
            print(f"│  💡 Collection Action:")
            print(f"│     {missing.collection_action}")
        print(f"│")
        print(f"└─────────────────────────────────────")
        print()
    
    print("CONFIDENCE IMPACT:")
    print("-" * 80)
    print()
    breakdown = explorer.calculate_confidence_breakdown()
    print(f"Overall Confidence: {breakdown['overall_confidence']:.0%}")
    print(f"  (Would be: {breakdown['overall_confidence'] + 0.10 + 0.25:.0%} with all evidence)")
    print()


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_evidence_explorer()
    demonstrate_evidence_card_details()
    demonstrate_contradiction_display()
    demonstrate_missing_data_display()
    
    print("=" * 80)
    print("✅ EVIDENCE EXPLORER DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("  1. Every claim is traceable to evidence")
    print("  2. Evidence shows: metric name, value, baseline, deviation, source")
    print("  3. Missing data is never hidden - it's explicitly shown")
    print("  4. Contradictions are displayed with resolution paths")
    print("  5. Confidence is transparent with component breakdown")
    print()
    print("A senior DBA can now say: 'This system is not guessing.'")
    print()

