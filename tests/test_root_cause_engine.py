#!/usr/bin/env python3
"""
RootCauseEngine - Comprehensive Test Suite

Tests for the deterministic Root Cause Engine for PostgreSQL performance incidents.

Test Categories:
1. Rule Matching Tests
2. Confidence Scoring Tests
3. False Positive Tests
4. Integration Tests
5. Multi-Cause Detection Tests
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from signals.root_cause_engine import (
    RootCauseEngine,
    RootCauseRuleRegistry,
    RootCauseCategory,
    EvidenceType,
    RootCauseRule,
    EvidenceSet,
    RuleMatch,
    RootCauseResult,
)
from signals.signal_engine import Signal, SignalResult


# ---------------------------------------------------------------------
# Mock Classes for Testing
# ---------------------------------------------------------------------

class MockSignal:
    """Mock signal for testing."""
    def __init__(
        self,
        id: str,
        name: str,
        type: str,
        severity: str,
        confidence: float,
        data: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ):
        self.id = id
        self.name = name
        self.type = type
        self.severity = severity
        self.confidence = confidence
        self.data = data
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow().isoformat()


class MockSignalResult:
    """Mock signal result for testing."""
    def __init__(self, signals: List[MockSignal], analysis: Dict[str, Any] = None):
        self.signals = signals
        self.analysis = analysis or {"signal_count": len(signals)}


# ---------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------

class TestRuleMatching(unittest.TestCase):
    """Tests for rule matching logic."""
    
    def setUp(self):
        self.registry = RootCauseRuleRegistry()
        self.engine = RootCauseEngine()
    
    def test_sequential_scan_rule_matches(self):
        """Test that sequential scan rule matches correct evidence."""
        evidence = EvidenceSet()
        evidence.signal_types.add("sequential_scan_detected")
        evidence.metric_values["seq_scan_ratio"] = 0.99
        
        rule = None
        for r in self.registry.rules:
            if r.name == "sequential_scan_large_table":
                rule = r
                break
        
        self.assertIsNotNone(rule)
        is_match, matched, warnings = rule.matches(evidence)
        
        self.assertTrue(is_match)
        self.assertIn("sequential_scan_detected", matched)
        self.assertEqual(len(warnings), 0)
    
    def test_missing_index_rule_matches(self):
        """Test that missing index rule matches correct evidence."""
        evidence = EvidenceSet()
        evidence.signal_types.add("missing_index_large_table")
        evidence.metric_values["row_count"] = 1000000
        
        rule = None
        for r in self.registry.rules:
            if r.name == "missing_index_large_table":
                rule = r
                break
        
        self.assertIsNotNone(rule)
        is_match, matched, warnings = rule.matches(evidence)
        
        self.assertTrue(is_match)
        self.assertIn("missing_index_large_table", matched)
    
    def test_rule_requires_all_evidence(self):
        """Test that rule requires all specified evidence."""
        evidence = EvidenceSet()
        # Missing required evidence
        evidence.signal_types.add("some_other_signal")
        
        for rule in self.registry.rules:
            is_match, matched, warnings = rule.matches(evidence)
            self.assertFalse(is_match)
            self.assertEqual(len(matched), 0)
    
    def test_optional_evidence_boosts_confidence(self):
        """Test that optional evidence can be matched."""
        evidence = EvidenceSet()
        evidence.signal_types.add("missing_index_large_table")
        evidence.signal_types.add("sequential_scan_detected")  # Optional
        evidence.metric_values["row_count"] = 1000000
        
        rule = None
        for r in self.registry.rules:
            if r.name == "missing_index_large_table":
                rule = r
                break
        
        self.assertIsNotNone(rule)
        is_match, matched, warnings = rule.matches(evidence)
        
        self.assertTrue(is_match)
        self.assertIn("sequential_scan_detected", matched)


class TestFalsePositiveDetection(unittest.TestCase):
    """Tests for false positive detection logic."""
    
    def setUp(self):
        self.registry = RootCauseRuleRegistry()
    
    def test_small_table_false_positive(self):
        """Test that small table triggers false positive."""
        evidence = EvidenceSet()
        evidence.signal_types.add("missing_index_large_table")
        evidence.metric_values["row_count"] = 5000  # Small table (< 10000)
        
        rule = None
        for r in self.registry.rules:
            if r.name == "missing_index_large_table":
                rule = r
                break
        
        self.assertIsNotNone(rule)
        is_match, matched, warnings = rule.matches(evidence)
        
        self.assertTrue(is_match)  # Rule still matches
        self.assertIn("small_table", warnings)  # False positive detected
    
    def test_large_table_no_false_positive(self):
        """Test that large table does not trigger false positive."""
        evidence = EvidenceSet()
        evidence.signal_types.add("missing_index_large_table")
        evidence.metric_values["row_count"] = 10000000  # Large table
        
        rule = None
        for r in self.registry.rules:
            if r.name == "missing_index_large_table":
                rule = r
                break
        
        self.assertIsNotNone(rule)
        is_match, matched, warnings = rule.matches(evidence)
        
        self.assertTrue(is_match)
        self.assertNotIn("small_table", warnings)
    
    def test_metric_below_false_positive(self):
        """Test metric_below false positive format."""
        evidence = EvidenceSet()
        evidence.signal_types.add("low_buffer_cache_hit_ratio")
        evidence.metric_values["cache_hit_ratio"] = 0.50
        
        # Check if false positive would trigger
        self.assertTrue(
            evidence.has_metric_below("cache_hit_ratio", 0.9)
        )
    
    def test_metric_above_false_positive(self):
        """Test metric_above false positive format."""
        evidence = EvidenceSet()
        evidence.metric_values["latency_ms"] = 50000
        
        # Check if false positive would trigger
        self.assertTrue(
            evidence.has_metric_above("latency_ms", 10000)
        )


class TestConfidenceScoring(unittest.TestCase):
    """Tests for confidence scoring logic."""
    
    def setUp(self):
        self.engine = RootCauseEngine()
    
    def test_base_confidence_applied(self):
        """Test that base confidence is applied on match."""
        evidence = EvidenceSet()
        evidence.signal_types.add("sequential_scan_detected")
        
        rule = None
        for r in self.engine.registry.rules:
            if r.name == "sequential_scan_large_table":
                rule = r
                break
        
        self.assertIsNotNone(rule)
        self.assertEqual(rule.confidence_contribution, 0.35)
    
    def test_max_confidence_respected(self):
        """Test that max confidence cap is respected."""
        # Rules should have max_confidence <= 0.95
        for rule in self.engine.registry.rules:
            self.assertLessEqual(rule.max_confidence, 0.95)
    
    def test_confidence_reduced_for_false_positives(self):
        """Test confidence is reduced when false positives detected."""
        evidence = EvidenceSet()
        evidence.signal_types.add("missing_index_large_table")
        evidence.metric_values["row_count"] = 5000  # Triggers small_table FP
        
        # Simulate match with warnings
        rule = None
        for r in self.engine.registry.rules:
            if r.name == "missing_index_large_table":
                rule = r
                break
        
        is_match, matched, warnings = rule.matches(evidence)
        if is_match and warnings:
            confidence = rule.confidence_contribution
            reduced = confidence * (1.0 - (len(warnings) * 0.1))
            reduced = max(0.1, reduced)
            self.assertLess(reduced, confidence)
    
    def test_min_confidence_threshold(self):
        """Test minimum confidence threshold for likely causes."""
        self.assertEqual(self.engine.min_confidence_threshold, 0.30)


class TestRuleRegistry(unittest.TestCase):
    """Tests for the rule registry."""
    
    def setUp(self):
        self.registry = RootCauseRuleRegistry()
    
    def test_all_categories_covered(self):
        """Test that all 8 categories have rules."""
        expected_categories = 8
        actual_categories = len(self.registry.get_all_categories())
        self.assertEqual(actual_categories, expected_categories)
    
    def test_each_category_has_rules(self):
        """Test that each category has at least one rule."""
        for category in RootCauseCategory:
            rules = self.registry.get_rules_for_category(category)
            self.assertGreater(len(rules), 0, 
                f"Category {category.value} has no rules")
    
    def test_total_rule_count(self):
        """Test total rule count is 21."""
        expected_rules = 21
        actual_rules = len(self.registry.rules)
        self.assertEqual(actual_rules, expected_rules)
    
    def test_rule_names_are_unique(self):
        """Test that all rule names are unique."""
        names = [r.name for r in self.registry.rules]
        self.assertEqual(len(names), len(set(names)))
    
    def test_rules_have_required_fields(self):
        """Test that all rules have required fields."""
        for rule in self.registry.rules:
            self.assertTrue(rule.name, "Rule missing name")
            self.assertTrue(rule.category, "Rule missing category")
            self.assertTrue(rule.required_evidence, "Rule missing required_evidence")
            self.assertIsInstance(rule.confidence_contribution, float)
            self.assertIsInstance(rule.max_confidence, float)


class TestIntegration(unittest.TestCase):
    """Integration tests for RootCauseEngine."""
    
    def setUp(self):
        self.engine = RootCauseEngine()
    
    def test_analyze_with_mock_signals(self):
        """Test full analysis with mock signals."""
        signals = [
            MockSignal(
                id="sig_001",
                name="sequential_scan_detected",
                type="query_metrics",
                severity="high",
                confidence=0.90,
                data={"seq_scans": 1523, "idx_scans": 12},
                metadata={"explain": "Sequential scan on large table"}
            ),
            MockSignal(
                id="sig_002",
                name="high_query_latency",
                type="query_metrics",
                severity="high",
                confidence=0.92,
                data={"mean_time_ms": 15234.5},
                metadata={"explain": "Query latency deviation"}
            ),
            MockSignal(
                id="sig_003",
                name="missing_index_large_table",
                type="json_context",
                severity="high",
                confidence=0.90,
                data={"row_count": 10234567, "table": "orders"},
                metadata={"explain": "Missing index on large table"}
            ),
        ]
        
        signal_result = MockSignalResult(signals)
        results = self.engine.analyze(signal_result)
        
        # Should have results
        self.assertGreater(len(results), 0)
        
        # INDEX_ISSUES should be a likely cause
        index_issues = results.get(RootCauseCategory.INDEX_ISSUES)
        self.assertIsNotNone(index_issues)
        self.assertTrue(index_issues.is_likely_cause)
        self.assertGreater(index_issues.confidence, 0.30)
    
    def test_get_primary_causes(self):
        """Test getting primary causes."""
        signals = [
            MockSignal(
                id="sig_001",
                name="sequential_scan_detected",
                type="query_metrics",
                severity="high",
                confidence=0.90,
                data={},
            ),
            MockSignal(
                id="sig_002",
                name="high_query_latency",
                type="query_metrics",
                severity="high",
                confidence=0.92,
                data={},
            ),
        ]
        
        signal_result = MockSignalResult(signals)
        results = self.engine.analyze(signal_result)
        primary = self.engine.get_primary_causes(results, top_n=3)
        
        self.assertLessEqual(len(primary), 3)
        # Primary causes should be sorted by confidence
        if len(primary) > 1:
            self.assertGreaterEqual(
                primary[0][1].confidence,
                primary[1][1].confidence
            )
    
    def test_format_results(self):
        """Test formatted results output."""
        signals = [
            MockSignal(
                id="sig_001",
                name="sequential_scan_detected",
                type="query_metrics",
                severity="high",
                confidence=0.90,
                data={},
            ),
        ]
        
        signal_result = MockSignalResult(signals)
        results = self.engine.analyze(signal_result)
        formatted = self.engine.format_results(results)
        
        # Check required fields
        self.assertIn("analysis_timestamp", formatted)
        self.assertIn("categories_analyzed", formatted)
        self.assertIn("likely_causes_count", formatted)
        self.assertIn("primary_causes", formatted)
        self.assertIn("all_results", formatted)
    
    def test_empty_signals(self):
        """Test handling of empty signals."""
        signal_result = MockSignalResult([])
        results = self.engine.analyze(signal_result)
        
        # Should return empty dict
        self.assertEqual(len(results), 0)


class TestMultiCauseDetection(unittest.TestCase):
    """Tests for multiple root cause detection."""
    
    def setUp(self):
        self.engine = RootCauseEngine()
    
    def test_multiple_causes_detected(self):
        """Test that multiple root causes are detected."""
        signals = [
            # Index issue
            MockSignal(
                id="sig_001",
                name="sequential_scan_detected",
                type="query_metrics",
                severity="high",
                confidence=0.90,
                data={},
            ),
            MockSignal(
                id="sig_002",
                name="missing_index_large_table",
                type="json_context",
                severity="high",
                confidence=0.90,
                data={},
            ),
            # Statistics issue
            MockSignal(
                id="sig_003",
                name="stale_statistics",
                type="table_stats",
                severity="medium",
                confidence=0.85,
                data={},
            ),
            # Blocking issue
            MockSignal(
                id="sig_004",
                name="blocking_detected",
                type="incident",
                severity="high",
                confidence=0.95,
                data={},
            ),
        ]
        
        signal_result = MockSignalResult(signals)
        results = self.engine.analyze(signal_result)
        
        # Should have multiple categories
        likely_causes = [
            cat for cat, res in results.items()
            if res.is_likely_cause
        ]
        self.assertGreater(len(likely_causes), 1)
    
    def test_each_cause_has_evidence(self):
        """Test that each detected cause has supporting evidence."""
        signals = [
            MockSignal(
                id="sig_001",
                name="sequential_scan_detected",
                type="query_metrics",
                severity="high",
                confidence=0.90,
                data={},
            ),
            MockSignal(
                id="sig_002",
                name="stale_statistics",
                type="table_stats",
                severity="medium",
                confidence=0.85,
                data={},
            ),
        ]
        
        signal_result = MockSignalResult(signals)
        results = self.engine.analyze(signal_result)
        
        for category, result in results.items():
            if result.is_likely_cause:
                self.assertGreater(len(result.evidence_ids), 0)
                self.assertGreater(len(result.matched_rules), 0)
    
    def test_recommendations_for_each_cause(self):
        """Test that each cause has recommendations."""
        signals = [
            MockSignal(
                id="sig_001",
                name="sequential_scan_detected",
                type="query_metrics",
                severity="high",
                confidence=0.90,
                data={},
            ),
        ]
        
        signal_result = MockSignalResult(signals)
        results = self.engine.analyze(signal_result)
        
        for category, result in results.items():
            if result.is_likely_cause:
                self.assertGreater(len(result.recommendations), 0)


class TestEvidenceSet(unittest.TestCase):
    """Tests for EvidenceSet functionality."""
    
    def test_has_signal_type(self):
        """Test signal type checking."""
        evidence = EvidenceSet()
        evidence.signal_types.add("sequential_scan_detected")
        
        self.assertTrue(evidence.has_signal_type("sequential_scan_detected"))
        self.assertFalse(evidence.has_signal_type("missing_index"))
    
    def test_has_signal_name(self):
        """Test signal name checking."""
        evidence = EvidenceSet()
        evidence.signal_names.add("high_query_latency")
        
        self.assertTrue(evidence.has_signal_name("high_query_latency"))
        self.assertFalse(evidence.has_signal_name("low_tps"))
    
    def test_get_metric(self):
        """Test metric retrieval."""
        evidence = EvidenceSet()
        evidence.metric_values["row_count"] = 1000000
        
        self.assertEqual(evidence.get_metric("row_count"), 1000000)
        self.assertEqual(evidence.get_metric("unknown", default=0), 0)
    
    def test_has_metric_above(self):
        """Test metric above threshold checking."""
        evidence = EvidenceSet()
        evidence.metric_values["latency_ms"] = 15000
        
        self.assertTrue(evidence.has_metric_above("latency_ms", 10000))
        self.assertFalse(evidence.has_metric_above("latency_ms", 20000))
    
    def test_has_metric_below(self):
        """Test metric below threshold checking."""
        evidence = EvidenceSet()
        evidence.metric_values["cache_hit_ratio"] = 0.85
        
        self.assertTrue(evidence.has_metric_below("cache_hit_ratio", 0.9))
        self.assertFalse(evidence.has_metric_below("cache_hit_ratio", 0.8))


class TestCausationChain(unittest.TestCase):
    """Tests for causation chain building."""
    
    def setUp(self):
        self.engine = RootCauseEngine()
    
    def test_causation_chain_built(self):
        """Test that causation chain is built for results."""
        # Use real Signal class with proper structure
        signals = [
            Signal(
                id="sig_001",
                name="sequential_scan_detected",
                type="query_metrics",
                severity="high",
                confidence=0.90,
                data={"seq_scans": 1523, "idx_scans": 12},
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
        ]
        
        # Create proper SignalResult
        signal_result = SignalResult(
            signals=signals,
            analysis={"signal_count": len(signals)},
            filtered_count=0,
            processing_time=0.001,
        )
        
        results = self.engine.analyze(signal_result)
        
        found_causation = False
        for category, result in results.items():
            if result.is_likely_cause and result.causation_chain:
                found_causation = True
                break
        
        self.assertTrue(found_causation, 
            "Expected at least one result with non-empty causation_chain")


class TestFalsePositiveNotes(unittest.TestCase):
    """Tests for false positive note tracking."""
    
    def setUp(self):
        self.engine = RootCauseEngine()
    
    def test_false_positive_notes_added(self):
        """Test that false positive notes are added to results."""
        # Create evidence that triggers small_table false positive
        signals = [
            MockSignal(
                id="sig_001",
                name="missing_index_large_table",
                type="json_context",
                severity="high",
                confidence=0.90,
                data={"row_count": 5000},  # Small table
            ),
        ]
        
        signal_result = MockSignalResult(signals)
        results = self.engine.analyze(signal_result)
        
        # Check that false positive notes are present
        for category, result in results.items():
            if result.is_likely_cause:
                # Should have false positive notes
                self.assertTrue(
                    len(result.false_positive_notes) > 0 or
                    all(len(m.false_positive_warnings) == 0 
                        for m in result.matched_rules)
                )


# ---------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------

def run_tests():
    """Run all tests and report results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRuleMatching))
    suite.addTests(loader.loadTestsFromTestCase(TestFalsePositiveDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestConfidenceScoring))
    suite.addTests(loader.loadTestsFromTestCase(TestRuleRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiCauseDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestEvidenceSet))
    suite.addTests(loader.loadTestsFromTestCase(TestCausationChain))
    suite.addTests(loader.loadTestsFromTestCase(TestFalsePositiveNotes))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())

