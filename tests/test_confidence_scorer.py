#!/usr/bin/env python3
"""
Test Suite: ConfidenceScorer for PostgreSQL Incidents

Tests for:
1. Mathematical model correctness
2. Boundary conditions
3. Edge cases
4. Determinism
5. Integration scenarios
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from signals.confidence_scorer import (
    ConfidenceScorer,
    SignalEvidence,
    ExpectedEvidence,
    ConfidenceBreakdown,
    calculate_incident_confidence,
)


# ---------------------------------------------------------------------
# Test Data Factory
# ---------------------------------------------------------------------

class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_signal(
        signal_id: str = "sig_001",
        signal_name: str = "test_signal",
        signal_type: str = "query_metrics",
        confidence: float = 0.85,
        data_timestamp: str = None,
    ) -> SignalEvidence:
        """Create a test signal."""
        return SignalEvidence(
            signal_id=signal_id,
            signal_name=signal_name,
            signal_type=signal_type,
            confidence=confidence,
            evidence_ids=[f"ev_{signal_id}"],
            data_timestamp=data_timestamp or datetime.utcnow().isoformat() + "Z",
        )
    
    @staticmethod
    def create_expected_evidence(
        evidence_type: str = "query_metrics",
        is_required: bool = True,
    ) -> ExpectedEvidence:
        """Create expected evidence."""
        return ExpectedEvidence(
            evidence_type=evidence_type,
            is_required=is_required,
        )


# ---------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------

class TestBaseConfidenceCalculation(unittest.TestCase):
    """Tests for base confidence calculation."""
    
    def setUp(self):
        self.scorer = ConfidenceScorer()
    
    def test_single_signal(self):
        """Test base confidence with single signal."""
        signals = [TestDataFactory.create_signal(confidence=0.90)]
        expected = [TestDataFactory.create_expected_evidence()]
        
        score, breakdown = self.scorer.calculate_confidence(signals, expected)
        
        # Base should equal confidence × sqrt(confidence)
        expected_base = 0.90 * (0.90 ** 0.5)
        self.assertAlmostEqual(breakdown.base_confidence, expected_base, places=4)
    
    def test_multiple_signals(self):
        """Test base confidence with multiple signals."""
        signals = [
            TestDataFactory.create_signal(signal_id="sig_001", confidence=0.90),
            TestDataFactory.create_signal(signal_id="sig_002", confidence=0.80),
            TestDataFactory.create_signal(signal_id="sig_003", confidence=0.70),
        ]
        expected = [TestDataFactory.create_expected_evidence()]
        
        score, breakdown = self.scorer.calculate_confidence(signals, expected)
        
        # Verify counts
        self.assertEqual(breakdown.signal_count, 3)
        self.assertAlmostEqual(breakdown.mean_confidence, 0.80, places=2)
        self.assertEqual(breakdown.min_confidence, 0.70)
    
    def test_lowest_confidence_dominates(self):
        """Test that lowest confidence has significant impact."""
        signals_high = [
            TestDataFactory.create_signal(signal_id="sig_001", confidence=0.99),
            TestDataFactory.create_signal(signal_id="sig_002", confidence=0.99),
        ]
        signals_mixed = [
            TestDataFactory.create_signal(signal_id="sig_001", confidence=0.99),
            TestDataFactory.create_signal(signal_id="sig_002", confidence=0.50),
        ]
        expected = [TestDataFactory.create_expected_evidence()]
        
        score_high, _ = self.scorer.calculate_confidence(signals_high, expected)
        score_mixed, _ = self.scorer.calculate_confidence(signals_mixed, expected)
        
        # Mixed should be significantly lower than all high
        self.assertLess(score_mixed, score_high * 0.8)
    
    def test_empty_signals(self):
        """Test handling of empty signals."""
        expected = [TestDataFactory.create_expected_evidence()]
        
        score, breakdown = self.scorer.calculate_confidence([], expected)
        
        self.assertEqual(score, 0.0)
        self.assertEqual(breakdown.signal_count, 0)


class TestEvidenceCompleteness(unittest.TestCase):
    """Tests for evidence completeness calculation."""
    
    def setUp(self):
        self.scorer = ConfidenceScorer()
    
    def test_all_evidence_present(self):
        """Test with all expected evidence present."""
        signals = [
            TestDataFactory.create_signal(signal_type="query_metrics"),
            TestDataFactory.create_signal(signal_type="table_stats"),
        ]
        expected = [
            TestDataFactory.create_expected_evidence("query_metrics", is_required=True),
            TestDataFactory.create_expected_evidence("table_stats", is_required=True),
        ]
        
        score, breakdown = self.scorer.calculate_confidence(signals, expected)
        
        self.assertEqual(breakdown.missing_evidence_count, 0)
        self.assertAlmostEqual(breakdown.evidence_completeness, 1.0, places=4)
        self.assertEqual(breakdown.completeness_penalty, 0.0)
    
    def test_missing_evidence_penalized(self):
        """Test that missing evidence reduces completeness."""
        signals = [TestDataFactory.create_signal(signal_type="query_metrics")]
        expected = [
            TestDataFactory.create_expected_evidence("query_metrics", is_required=True),
            TestDataFactory.create_expected_evidence("table_stats", is_required=True),
            TestDataFactory.create_expected_evidence("index_health", is_required=True),
        ]
        
        score, breakdown = self.scorer.calculate_confidence(signals, expected)
        
        self.assertEqual(breakdown.missing_evidence_count, 2)
        self.assertAlmostEqual(breakdown.evidence_completeness, 0.80, places=4)
        self.assertAlmostEqual(breakdown.completeness_penalty, 0.20, places=4)
    
    def test_optional_evidence_not_penalized(self):
        """Test that missing optional evidence doesn't penalize."""
        signals = [TestDataFactory.create_signal(signal_type="query_metrics")]
        expected = [
            TestDataFactory.create_expected_evidence("query_metrics", is_required=True),
            TestDataFactory.create_expected_evidence("table_stats", is_required=False),  # Optional
        ]
        
        score, breakdown = self.scorer.calculate_confidence(signals, expected)
        
        self.assertEqual(breakdown.missing_evidence_count, 0)
        self.assertAlmostEqual(breakdown.evidence_completeness, 1.0, places=4)
    
    def test_min_completeness(self):
        """Test minimum completeness is enforced."""
        signals = []
        expected = [
            TestDataFactory.create_expected_evidence("query_metrics", is_required=True),
            TestDataFactory.create_expected_evidence("table_stats", is_required=True),
            TestDataFactory.create_expected_evidence("index_health", is_required=True),
            TestDataFactory.create_expected_evidence("config", is_required=True),
            TestDataFactory.create_expected_evidence("logs", is_required=True),
            TestDataFactory.create_expected_evidence("plans", is_required=True),
        ]
        
        score, breakdown = self.scorer.calculate_confidence(signals, expected)
        
        # Should be capped at 0.5
        self.assertAlmostEqual(breakdown.evidence_completeness, 0.5, places=4)
    
    def test_no_expected_evidence(self):
        """Test with no expected evidence."""
        signals = [TestDataFactory.create_signal()]
        
        score, breakdown = self.scorer.calculate_confidence(signals, [])
        
        self.assertEqual(breakdown.missing_evidence_count, 0)
        self.assertAlmostEqual(breakdown.evidence_completeness, 1.0, places=4)


class TestSignalAgreement(unittest.TestCase):
    """Tests for signal agreement boost."""
    
    def setUp(self):
        self.scorer = ConfidenceScorer()
    
    def test_single_signal_group(self):
        """Test with single signal type (no boost)."""
        signals = [
            TestDataFactory.create_signal(signal_type="query_metrics"),
            TestDataFactory.create_signal(signal_type="query_metrics"),
            TestDataFactory.create_signal(signal_type="query_metrics"),
        ]
        expected = [TestDataFactory.create_expected_evidence()]
        
        score, breakdown = self.scorer.calculate_confidence(signals, expected)
        
        self.assertEqual(breakdown.independent_signal_groups, 1)
        self.assertAlmostEqual(breakdown.signal_agreement, 1.0, places=4)
        self.assertEqual(breakdown.agreement_boost, 0.0)
    
    def test_multiple_signal_groups(self):
        """Test with multiple signal types (boost)."""
        signals = [
            TestDataFactory.create_signal(signal_id="sig_001", signal_type="query_metrics"),
            TestDataFactory.create_signal(signal_id="sig_002", signal_type="table_stats"),
            TestDataFactory.create_signal(signal_id="sig_003", signal_type="query_metrics"),
        ]
        expected = [TestDataFactory.create_expected_evidence()]
        
        score, breakdown = self.scorer.calculate_confidence(signals, expected)
        
        self.assertEqual(breakdown.independent_signal_groups, 2)
        self.assertAlmostEqual(breakdown.signal_agreement, 1.05, places=4)
        self.assertAlmostEqual(breakdown.agreement_boost, 0.05, places=4)
    
    def test_max_agreement_boost(self):
        """Test maximum agreement boost is capped."""
        # Create 6 signals with different types
        signals = [
            TestDataFactory.create_signal(signal_id=f"sig_{i}", signal_type=f"type_{i}")
            for i in range(6)
        ]
        expected = [TestDataFactory.create_expected_evidence()]
        
        score, breakdown = self.scorer.calculate_confidence(signals, expected)
        
        # Should be capped at 1.2 (20% boost)
        self.assertAlmostEqual(breakdown.signal_agreement, 1.20, places=4)
        self.assertAlmostEqual(breakdown.agreement_boost, 0.20, places=4)


class TestDataFreshness(unittest.TestCase):
    """Tests for data freshness penalty."""
    
    def setUp(self):
        self.scorer = ConfidenceScorer()
    
    def test_recent_data(self):
        """Test with recent data (minimal penalty)."""
        now = datetime.utcnow()
        signals = [
            TestDataFactory.create_signal(
                data_timestamp=(now - timedelta(minutes=15)).isoformat() + "Z"
            ),
        ]
        expected = [TestDataFactory.create_expected_evidence()]
        
        score, breakdown = self.scorer.calculate_confidence(signals, expected)
        
        self.assertLess(breakdown.max_data_age_hours, 1.0)
        self.assertGreater(breakdown.data_freshness, 0.95)
    
    def test_stale_data(self):
        """Test with stale data (penalty)."""
        now = datetime.utcnow()
        signals = [
            TestDataFactory.create_signal(
                data_timestamp=(now - timedelta(hours=10)).isoformat() + "Z"
            ),
        ]
        expected = [TestDataFactory.create_expected_evidence()]
        
        score, breakdown = self.scorer.calculate_confidence(signals, expected)
        
        self.assertGreater(breakdown.max_data_age_hours, 8.0)
        self.assertLess(breakdown.data_freshness, 0.9)
        self.assertGreater(breakdown.freshness_penalty, 0.1)
    
    def test_max_freshness_penalty(self):
        """Test maximum freshness penalty is capped."""
        now = datetime.utcnow()
        signals = [
            TestDataFactory.create_signal(
                data_timestamp=(now - timedelta(hours=20)).isoformat() + "Z"
            ),
        ]
        expected = [TestDataFactory.create_expected_evidence()]
        
        score, breakdown = self.scorer.calculate_confidence(signals, expected)
        
        # Should be capped at 0.7 (30% penalty)
        self.assertAlmostEqual(breakdown.data_freshness, 0.7, places=4)
    
    def test_uses_max_age(self):
        """Test that freshness uses the maximum age across signals."""
        now = datetime.utcnow()
        signals = [
            TestDataFactory.create_signal(
                signal_id="sig_001",
                data_timestamp=(now - timedelta(minutes=10)).isoformat() + "Z"
            ),
            TestDataFactory.create_signal(
                signal_id="sig_002",
                data_timestamp=(now - timedelta(hours=5)).isoformat() + "Z"
            ),
        ]
        expected = [TestDataFactory.create_expected_evidence()]
        
        score, breakdown = self.scorer.calculate_confidence(signals, expected)
        
        # Should use max age (5 hours)
        self.assertAlmostEqual(breakdown.max_data_age_hours, 5.0, places=1)


class TestConflictPenalty(unittest.TestCase):
    """Tests for conflict penalty."""
    
    def setUp(self):
        self.scorer = ConfidenceScorer()
    
    def test_no_conflicts(self):
        """Test with no conflicts (no penalty)."""
        signals = [
            TestDataFactory.create_signal(signal_id="sig_001"),
            TestDataFactory.create_signal(signal_id="sig_002"),
        ]
        expected = [TestDataFactory.create_expected_evidence()]
        
        score, breakdown = self.scorer.calculate_confidence(signals, expected)
        
        self.assertEqual(breakdown.conflict_count, 0)
        self.assertAlmostEqual(breakdown.conflict_penalty, 1.0, places=4)
        self.assertEqual(breakdown.conflict_penalty_value, 0.0)
    
    def test_with_conflicts(self):
        """Test with conflicts (penalty)."""
        signals = [
            TestDataFactory.create_signal(signal_id="sig_001"),
            TestDataFactory.create_signal(signal_id="sig_002"),
        ]
        expected = [TestDataFactory.create_expected_evidence()]
        conflict_groups = [
            ["sig_001", "sig_002"]  # These two conflict
        ]
        
        score, breakdown = self.scorer.calculate_confidence(
            signals, expected, conflict_groups=conflict_groups
        )
        
        self.assertEqual(breakdown.conflict_count, 1)
        self.assertAlmostEqual(breakdown.conflict_penalty, 0.85, places=4)
        self.assertAlmostEqual(breakdown.conflict_penalty_value, 0.15, places=4)
    
    def test_multiple_conflicts(self):
        """Test with multiple conflict groups."""
        signals = [
            TestDataFactory.create_signal(signal_id="sig_001"),
            TestDataFactory.create_signal(signal_id="sig_002"),
            TestDataFactory.create_signal(signal_id="sig_003"),
        ]
        expected = [TestDataFactory.create_expected_evidence()]
        conflict_groups = [
            ["sig_001", "sig_002"],
            ["sig_002", "sig_003"],
        ]
        
        score, breakdown = self.scorer.calculate_confidence(
            signals, expected, conflict_groups=conflict_groups
        )
        
        self.assertEqual(breakdown.conflict_count, 2)
        self.assertAlmostEqual(breakdown.conflict_penalty, 0.70, places=4)
    
    def test_min_conflict_penalty(self):
        """Test minimum conflict penalty is enforced."""
        signals = [
            TestDataFactory.create_signal(signal_id=f"sig_{i}") for i in range(5)
        ]
        expected = [TestDataFactory.create_expected_evidence()]
        conflict_groups = [
            ["sig_001", "sig_002"],
            ["sig_002", "sig_003"],
            ["sig_003", "sig_004"],
            ["sig_004", "sig_005"],
        ]
        
        score, breakdown = self.scorer.calculate_confidence(
            signals, expected, conflict_groups=conflict_groups
        )
        
        # Should be capped at 0.4 (60% penalty)
        self.assertAlmostEqual(breakdown.conflict_penalty, 0.4, places=4)


class TestScoreBoundaries(unittest.TestCase):
    """Tests for score boundary conditions."""
    
    def setUp(self):
        self.scorer = ConfidenceScorer()
    
    def test_score_never_negative(self):
        """Test that score is never negative."""
        signals = [TestDataFactory.create_signal(confidence=0.1)]
        expected = [TestDataFactory.create_expected_evidence()]
        conflict_groups = [["sig_001", "sig_002"]] * 10  # Many conflicts
        
        score, _ = self.scorer.calculate_confidence(
            signals, expected, conflict_groups=conflict_groups
        )
        
        self.assertGreaterEqual(score, 0.0)
    
    def test_score_never_exceeds_one(self):
        """Test that score is never greater than 1."""
        signals = [
            TestDataFactory.create_signal(confidence=0.99),
            TestDataFactory.create_signal(confidence=0.99),
            TestDataFactory.create_signal(confidence=0.99),
        ]
        expected = [TestDataFactory.create_expected_evidence()]
        
        score, _ = self.scorer.calculate_confidence(signals, expected)
        
        self.assertLessEqual(score, 1.0)
    
    def test_max_confidence_scenario(self):
        """Test maximum possible confidence scenario."""
        signals = [
            TestDataFactory.create_signal(
                signal_id=f"sig_{i}",
                signal_type=f"type_{i % 5}",  # 5 independent groups
                confidence=0.99,
                data_timestamp=datetime.utcnow().isoformat() + "Z"
            )
            for i in range(10)
        ]
        expected = [
            TestDataFactory.create_expected_evidence(f"type_{i}", is_required=True)
            for i in range(5)
        ]
        
        score, breakdown = self.scorer.calculate_confidence(signals, expected)
        
        # Should be very high but not exceed 1.0
        self.assertGreater(score, 0.8)
        self.assertLessEqual(score, 1.0)


class TestDeterminism(unittest.TestCase):
    """Tests for deterministic behavior."""
    
    def test_same_inputs_same_output(self):
        """Test that same inputs produce same outputs."""
        # Use fixed timestamps to ensure determinism
        fixed_timestamp = "2024-01-25T10:00:00+00:00"
        
        scorer = ConfidenceScorer(seed=42)

        signals = [
            TestDataFactory.create_signal(
                signal_id="sig_001", 
                confidence=0.85,
                data_timestamp=fixed_timestamp
            ),
            TestDataFactory.create_signal(
                signal_id="sig_002", 
                confidence=0.90,
                data_timestamp=fixed_timestamp
            ),
        ]
        expected = [
            TestDataFactory.create_expected_evidence("query_metrics", is_required=True),
            TestDataFactory.create_expected_evidence("table_stats", is_required=True),
        ]

        scores = []
        for _ in range(5):
            score, _ = scorer.calculate_confidence(signals, expected)
            scores.append(score)

        # All scores should be identical (within floating point tolerance)
        first_score = scores[0]
        all_same = all(abs(s - first_score) < 1e-10 for s in scores)
        self.assertTrue(all_same, f"Scores were not identical: {scores}")
    
    def test_different_inputs_different_output(self):
        """Test that different inputs produce different outputs."""
        scorer = ConfidenceScorer(seed=42)
        
        # Scenario 1
        signals1 = [
            TestDataFactory.create_signal(confidence=0.90),
            TestDataFactory.create_signal(confidence=0.80),
        ]
        expected = [TestDataFactory.create_expected_evidence()]
        score1, _ = scorer.calculate_confidence(signals1, expected)
        
        # Scenario 2
        signals2 = [
            TestDataFactory.create_signal(confidence=0.50),
            TestDataFactory.create_signal(confidence=0.50),
        ]
        score2, _ = scorer.calculate_confidence(signals2, expected)
        
        self.assertNotEqual(score1, score2)


class TestExplainScore(unittest.TestCase):
    """Tests for score explanation functionality."""
    
    def setUp(self):
        self.scorer = ConfidenceScorer()
    
    def test_explanation_contains_required_fields(self):
        """Test that explanation contains all required fields."""
        signals = [TestDataFactory.create_signal()]
        expected = [TestDataFactory.create_expected_evidence()]
        
        score, breakdown = self.scorer.calculate_confidence(signals, expected)
        explanation = self.scorer.explain_score(score, breakdown)
        
        required_fields = ["score", "rating", "breakdown", "interpretation", "recommendations"]
        for field in required_fields:
            self.assertIn(field, explanation)
    
    def test_rating_boundaries(self):
        """Test that ratings are assigned correctly."""
        signals = [TestDataFactory.create_signal()]
        expected = [TestDataFactory.create_expected_evidence()]
        
        # Very high score
        signals_high = [TestDataFactory.create_signal(signal_id="sig_001", confidence=0.99)]
        score_high, breakdown_high = self.scorer.calculate_confidence(signals_high, expected)
        explanation_high = self.scorer.explain_score(score_high, breakdown_high)
        self.assertEqual(explanation_high["rating"], "Very High")
        
        # Very low score
        signals_low = [TestDataFactory.create_signal(signal_id="sig_001", confidence=0.01)]
        score_low, breakdown_low = self.scorer.calculate_confidence(signals_low, expected)
        explanation_low = self.scorer.explain_score(score_low, breakdown_low)
        self.assertEqual(explanation_low["rating"], "Very Low")


class TestConfidenceBreakdown(unittest.TestCase):
    """Tests for ConfidenceBreakdown functionality."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        breakdown = ConfidenceBreakdown(
            raw_score=0.75,
            base_confidence=0.85,
            evidence_completeness=0.9,
            signal_agreement=1.05,
            data_freshness=0.96,
            conflict_penalty=1.0,
        )
        
        result = breakdown.to_dict()
        
        self.assertIn("raw_score", result)
        self.assertIn("components", result)
        self.assertIn("component_details", result)
        self.assertIn("penalties_and_boosts", result)
    
    def test_to_human_readable(self):
        """Test human-readable output."""
        breakdown = ConfidenceBreakdown(
            raw_score=0.75,
            base_confidence=0.85,
            evidence_completeness=0.9,
            signal_agreement=1.05,
            data_freshness=0.96,
            conflict_penalty=1.0,
            signal_count=3,
            mean_confidence=0.85,
            min_confidence=0.80,
            missing_evidence_count=1,
            independent_signal_groups=2,
            max_data_age_hours=2.0,
            conflict_count=0,
            completeness_penalty=0.10,
            agreement_boost=0.05,
            freshness_penalty=0.04,
            conflict_penalty_value=0.0,
        )
        
        output = breakdown.to_human_readable()
        
        self.assertIn("75.00%", output)
        self.assertIn("Component Breakdown:", output)
        self.assertIn("Base Confidence:", output)
        self.assertIn("Independent groups: 2", output)


class TestConvenienceFunction(unittest.TestCase):
    """Tests for the convenience function."""
    
    def test_calculate_incident_confidence(self):
        """Test the convenience function."""
        score, explanation = calculate_incident_confidence(
            signal_confidences=[0.90, 0.85, 0.88],
            evidence_completeness={"query_metrics": True, "table_stats": True},
            signal_agreement_groups={"metrics": 2, "stats": 1},
            data_freshness_hours={"metrics": 1.0, "stats": 2.0},
            conflicting_evidence=[],
            seed=42,
        )
        
        self.assertIsInstance(score, float)
        self.assertIsInstance(explanation, dict)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestRealisticScenario(unittest.TestCase):
    """Test with realistic PostgreSQL incident scenario."""
    
    def setUp(self):
        self.scorer = ConfidenceScorer(seed=42)
    
    def test_slow_query_missing_index(self):
        """Test the slow query due to missing index scenario."""
        now = datetime.utcnow()
        
        signals = [
            SignalEvidence(
                signal_id="sig_seq_scan",
                signal_name="sequential_scan_detected",
                signal_type="query_metrics",
                confidence=0.90,
                data_timestamp=(now - timedelta(minutes=15)).isoformat() + "Z",
            ),
            SignalEvidence(
                signal_id="sig_latency",
                signal_name="high_query_latency",
                signal_type="query_metrics",
                confidence=0.92,
                data_timestamp=(now - timedelta(minutes=10)).isoformat() + "Z",
            ),
            SignalEvidence(
                signal_id="sig_missing_idx",
                signal_name="missing_index_large_table",
                signal_type="table_stats",
                confidence=0.88,
                data_timestamp=(now - timedelta(minutes=5)).isoformat() + "Z",
            ),
            SignalEvidence(
                signal_id="sig_row_est",
                signal_name="row_estimation_error",
                signal_type="query_metrics",
                confidence=0.85,
                data_timestamp=(now - timedelta(minutes=10)).isoformat() + "Z",
            ),
        ]
        
        expected = [
            ExpectedEvidence(evidence_type="query_metrics", is_required=True),
            ExpectedEvidence(evidence_type="table_stats", is_required=True),
            ExpectedEvidence(evidence_type="execution_plan", is_required=True),
            ExpectedEvidence(evidence_type="index_health", is_required=False),
        ]
        
        score, breakdown = self.scorer.calculate_confidence(signals, expected)
        
        # Verify realistic scenario produces reasonable confidence
        self.assertGreater(score, 0.6)
        self.assertLess(score, 0.95)
        
        # Verify components
        self.assertGreater(breakdown.base_confidence, 0.7)
        self.assertEqual(breakdown.independent_signal_groups, 2)  # query_metrics, table_stats
        self.assertLess(breakdown.max_data_age_hours, 1.0)


# ---------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------

def run_tests():
    """Run all tests and report results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBaseConfidenceCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestEvidenceCompleteness))
    suite.addTests(loader.loadTestsFromTestCase(TestSignalAgreement))
    suite.addTests(loader.loadTestsFromTestCase(TestDataFreshness))
    suite.addTests(loader.loadTestsFromTestCase(TestConflictPenalty))
    suite.addTests(loader.loadTestsFromTestCase(TestScoreBoundaries))
    suite.addTests(loader.loadTestsFromTestCase(TestDeterminism))
    suite.addTests(loader.loadTestsFromTestCase(TestExplainScore))
    suite.addTests(loader.loadTestsFromTestCase(TestConfidenceBreakdown))
    suite.addTests(loader.loadTestsFromTestCase(TestConvenienceFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestRealisticScenario))
    
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

