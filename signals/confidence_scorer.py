#!/usr/bin/env python3
"""
ConfidenceScorer - Deterministic Confidence Scoring Model for PostgreSQL Incidents

Mathematical Model:
    Confidence = Base × Completeness × Agreement × Freshness × Conflict_Penalty

Where:
    - Base: Geometric mean of signal confidences, weighted by minimum confidence
            Base = (∏ci)^(1/n) × min(ci)^0.5
    
    - Completeness: Penalty for missing evidence
            Completeness = 1.0 - (missing_evidence_count × 0.1)
            Min: 0.5 (when 5+ evidence items missing)
    
    - Agreement: Boost for multiple independent signals agreeing
            Agreement = 1.0 + min((independent_groups - 1) × 0.05, 0.2)
            Max boost: 0.2 (when 5+ independent groups agree)
    
    - Freshness: Penalty for stale data
            Freshness = 1.0 - min(hours_since_update × 0.02, 0.3)
            Max penalty: 0.3 (when data is 15+ hours old)
    
    - Conflict_Penalty: Penalty for conflicting evidence
            Conflict_Penalty = 1.0 - (conflict_count × 0.15)
            Min: 0.4 (when 4+ conflicts detected)

Design Principles:
- Deterministic: Same inputs always produce same outputs
- Explainable: Full breakdown of score components
- Penalizes: Missing data, stale evidence, conflicting signals
- Boosts: Multiple independent signals agreeing on same conclusion

Usage:
    scorer = ConfidenceScorer()
    score, breakdown = scorer.calculate_confidence(
        signal_confidences=[0.92, 0.88, 0.95],
        evidence_completeness={"query_metrics": True, "table_stats": True},
        signal_agreement_groups={"missing_index": 3},
        data_freshness_hours={"query_metrics": 0.5, "table_stats": 2.0},
        conflicting_evidence=[],
    )
"""

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------

@dataclass
class ConfidenceBreakdown:
    """
    Detailed breakdown of confidence score calculation.
    
    Provides full explainability for how the final score was computed.
    """
    # Raw values
    raw_score: float = 0.0
    base_confidence: float = 0.0
    evidence_completeness: float = 0.0
    signal_agreement: float = 0.0
    data_freshness: float = 0.0
    conflict_penalty: float = 1.0
    
    # Component details
    signal_count: int = 0
    mean_confidence: float = 0.0
    min_confidence: float = 0.0
    missing_evidence_count: int = 0
    independent_signal_groups: int = 0
    max_data_age_hours: float = 0.0
    conflict_count: int = 0
    
    # Penalties and boosts applied
    completeness_penalty: float = 0.0
    agreement_boost: float = 0.0
    freshness_penalty: float = 0.0
    conflict_penalty_value: float = 0.0
    
    # Metadata
    calculation_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    formula_used: str = "Confidence = Base × Completeness × Agreement × Freshness × Conflict_Penalty"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "raw_score": round(self.raw_score, 4),
            "components": {
                "base_confidence": round(self.base_confidence, 4),
                "evidence_completeness": round(self.evidence_completeness, 4),
                "signal_agreement": round(self.signal_agreement, 4),
                "data_freshness": round(self.data_freshness, 4),
                "conflict_penalty": round(self.conflict_penalty, 4),
            },
            "component_details": {
                "signal_count": self.signal_count,
                "mean_confidence": round(self.mean_confidence, 4),
                "min_confidence": round(self.min_confidence, 4),
                "missing_evidence_count": self.missing_evidence_count,
                "independent_signal_groups": self.independent_signal_groups,
                "max_data_age_hours": round(self.max_data_age_hours, 2),
                "conflict_count": self.conflict_count,
            },
            "penalties_and_boosts": {
                "completeness_penalty": round(self.completeness_penalty, 4),
                "agreement_boost": round(self.agreement_boost, 4),
                "freshness_penalty": round(self.freshness_penalty, 4),
                "conflict_penalty_value": round(self.conflict_penalty_value, 4),
            },
            "formula_used": self.formula_used,
            "calculation_timestamp": self.calculation_timestamp,
        }
    
    def to_human_readable(self) -> str:
        """Generate human-readable explanation of the score."""
        lines = [
            f"Confidence Score: {self.raw_score:.2%}",
            "",
            "Component Breakdown:",
            f"  Base Confidence: {self.base_confidence:.4f}",
            f"    - Signals: {self.signal_count}, Mean: {self.mean_confidence:.2f}, Min: {self.min_confidence:.2f}",
            f"  Evidence Completeness: {self.evidence_completeness:.4f}",
            f"    - Missing items: {self.missing_evidence_count} (penalty: -{self.completeness_penalty:.2f})",
            f"  Signal Agreement: {self.signal_agreement:.4f}",
            f"    - Independent groups: {self.independent_signal_groups} (boost: +{self.agreement_boost:.2f})",
            f"  Data Freshness: {self.data_freshness:.4f}",
            f"    - Max age: {self.max_data_age_hours:.1f}h (penalty: -{self.freshness_penalty:.2f})",
            f"  Conflict Penalty: {self.conflict_penalty:.4f}",
            f"    - Conflicts: {self.conflict_count} (penalty: -{self.conflict_penalty_value:.2f})",
            "",
            f"Formula: Confidence = {self.base_confidence:.4f} × {self.evidence_completeness:.4f} × "
            f"{self.signal_agreement:.4f} × {self.data_freshness:.4f} × {self.conflict_penalty:.4f}",
        ]
        return "\n".join(lines)


@dataclass
class SignalEvidence:
    """
    A signal with associated evidence for confidence scoring.
    
    Attributes:
        signal_id: Unique identifier for the signal
        signal_name: Human-readable name
        signal_type: Type category (query_metrics, table_stats, etc.)
        confidence: Individual signal confidence (0.0 to 1.0)
        evidence_ids: List of evidence IDs supporting this signal
        data_timestamp: When the data was collected
        metadata: Additional signal metadata
    """
    signal_id: str
    signal_name: str
    signal_type: str
    confidence: float
    evidence_ids: List[str] = field(default_factory=list)
    data_timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpectedEvidence:
    """
    Definition of expected evidence for an incident type.
    
    Used to calculate evidence completeness score.
    
    Attributes:
        evidence_type: Type of evidence expected
        is_required: Whether this evidence is required
        freshness_threshold_hours: Max age for this evidence type
    """
    evidence_type: str
    is_required: bool = True
    freshness_threshold_hours: float = 24.0


# ---------------------------------------------------------------------
# Confidence Scorer
# ---------------------------------------------------------------------

class ConfidenceScorer:
    """
    Deterministic confidence scorer for PostgreSQL incidents.
    
    Calculates a confidence score between 0.0 and 1.0 based on:
    - Signal confidences (individual signal quality)
    - Evidence completeness (how much expected evidence is present)
    - Signal agreement (multiple independent signals agreeing)
    - Data freshness (how recent the data is)
    - Conflicting evidence (signals that contradict each other)
    
    All calculations are deterministic and reproducible.
    """
    
    # Constants for scoring weights
    BASE_WEIGHT = 1.0
    MISSING_EVIDENCE_PENALTY = 0.1  # Per missing evidence item
    MIN_COMPLETENESS = 0.5  # Minimum completeness score
    AGREEMENT_BOOST_PER_GROUP = 0.05  # Per independent group
    MAX_AGREEMENT_BOOST = 0.2  # Maximum agreement boost
    FRESHNESS_PENALTY_PER_HOUR = 0.02  # Per hour of staleness
    MAX_FRESHNESS_PENALTY = 0.3  # Maximum freshness penalty
    CONFLICT_PENALTY_PER_CONFLICT = 0.15  # Per conflicting evidence
    MIN_CONFLICT_PENALTY = 0.4  # Minimum conflict penalty
    
    def __init__(
        self,
        seed: Optional[int] = None,
        log_level: str = "INFO",
    ):
        """
        Initialize the confidence scorer.
        
        Args:
            seed: Optional random seed for deterministic behavior
            log_level: Logging level
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        
        # Set deterministic seed for any future randomness
        if seed is not None:
            random.seed(seed)
        
        self._seed = seed
    
    def calculate_confidence(
        self,
        signals: List[SignalEvidence],
        expected_evidence: List[ExpectedEvidence],
        conflict_groups: Optional[List[List[str]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, ConfidenceBreakdown]:
        """
        Calculate confidence score for an incident.
        
        Args:
            signals: List of signals with their evidence
            expected_evidence: List of expected evidence types
            conflict_groups: Groups of signals that conflict with each other
            context: Additional context for scoring
            
        Returns:
            Tuple of (confidence_score, ConfidenceBreakdown)
        """
        self.logger.info("Calculating confidence for %d signals", len(signals))
        
        # Initialize breakdown
        breakdown = ConfidenceBreakdown()
        conflict_groups = conflict_groups or []
        
        # 1. Calculate base confidence from signals
        base_confidence = self._calculate_base_confidence(signals, breakdown)
        breakdown.base_confidence = base_confidence
        
        # 2. Calculate evidence completeness
        completeness = self._calculate_evidence_completeness(
            signals, expected_evidence, breakdown
        )
        breakdown.evidence_completeness = completeness
        
        # 3. Calculate signal agreement boost
        agreement = self._calculate_signal_agreement(signals, breakdown)
        breakdown.signal_agreement = agreement
        
        # 4. Calculate data freshness penalty
        freshness = self._calculate_data_freshness(signals, breakdown)
        breakdown.data_freshness = freshness
        
        # 5. Calculate conflict penalty
        conflict_penalty = self._calculate_conflict_penalty(
            signals, conflict_groups, breakdown
        )
        breakdown.conflict_penalty = conflict_penalty
        
        # 6. Calculate final score
        # Confidence = Base × Completeness × Agreement × Freshness × Conflict_Penalty
        raw_score = (
            base_confidence
            * completeness
            * agreement
            * freshness
            * conflict_penalty
        )
        
        # Ensure score is within bounds
        raw_score = max(0.0, min(1.0, raw_score))
        breakdown.raw_score = raw_score
        
        self.logger.info(
            "Confidence calculated: %.4f (Base: %.4f, Comp: %.4f, Agree: %.4f, Fresh: %.4f, Conflict: %.4f)",
            raw_score, base_confidence, completeness, agreement, freshness, conflict_penalty
        )
        
        return raw_score, breakdown
    
    def _calculate_base_confidence(
        self,
        signals: List[SignalEvidence],
        breakdown: ConfidenceBreakdown,
    ) -> float:
        """
        Calculate base confidence from signal confidences.
        
        Uses geometric mean weighted by minimum confidence:
        Base = (∏ci)^(1/n) × min(ci)^0.5
        
        This gives more weight to the lowest confidence signal,
        as the overall confidence is limited by the weakest signal.
        """
        if not signals:
            breakdown.signal_count = 0
            breakdown.mean_confidence = 0.0
            breakdown.min_confidence = 0.0
            return 0.0
        
        confidences = [s.confidence for s in signals]
        n = len(confidences)
        
        breakdown.signal_count = n
        breakdown.mean_confidence = sum(confidences) / n
        breakdown.min_confidence = min(confidences)
        
        # Geometric mean
        try:
            geom_mean = math.pow(math.prod(confidences), 1.0 / n)
        except (ValueError, OverflowError):
            # Handle edge case where product is 0
            geom_mean = 0.0
        
        # Weighted by square root of minimum confidence
        # This ensures low-confidence signals have significant impact
        weighted_base = geom_mean * math.sqrt(breakdown.min_confidence)
        
        self.logger.debug(
            "Base confidence: geom_mean=%.4f, min=%.4f, weighted=%.4f",
            geom_mean, breakdown.min_confidence, weighted_base
        )
        
        return weighted_base
    
    def _calculate_evidence_completeness(
        self,
        signals: List[SignalEvidence],
        expected_evidence: List[ExpectedEvidence],
        breakdown: ConfidenceBreakdown,
    ) -> float:
        """
        Calculate evidence completeness score.
        
        Completeness = 1.0 - (missing_evidence_count × 0.1)
        Min Completeness = 0.5 (when 5+ evidence items missing)
        
        Only counts missing REQUIRED evidence.
        """
        if not expected_evidence:
            breakdown.missing_evidence_count = 0
            breakdown.completeness_penalty = 0.0
            return 1.0
        
        # Build set of present evidence types from signals
        present_types = set()
        for signal in signals:
            present_types.add(signal.signal_type)
            # Also check evidence_ids for additional types
            for ev_id in signal.evidence_ids:
                # Extract type from evidence ID if possible
                if "_" in ev_id:
                    ev_type = ev_id.split("_")[0]
                    present_types.add(ev_type)
        
        # Count missing required evidence
        missing_count = 0
        for expected in expected_evidence:
            if expected.is_required:
                # Check if we have this evidence type
                found = any(expected.evidence_type in pt for pt in present_types)
                if not found:
                    missing_count += 1
        
        breakdown.missing_evidence_count = missing_count
        
        # Calculate completeness
        completeness = 1.0 - (missing_count * self.MISSING_EVIDENCE_PENALTY)
        completeness = max(self.MIN_COMPLETENESS, completeness)
        
        breakdown.completeness_penalty = 1.0 - completeness
        
        self.logger.debug(
            "Evidence completeness: missing=%d, completeness=%.4f",
            missing_count, completeness
        )
        
        return completeness
    
    def _calculate_signal_agreement(
        self,
        signals: List[SignalEvidence],
        breakdown: ConfidenceBreakdown,
    ) -> float:
        """
        Calculate signal agreement boost.
        
        When multiple independent signals point to the same conclusion,
        confidence increases.
        
        Agreement = 1.0 + min((independent_groups - 1) × 0.05, 0.2)
        Max boost = 0.2 (when 5+ independent groups agree)
        
        Signals are grouped by their signal_type to identify independence.
        """
        if len(signals) <= 1:
            breakdown.independent_signal_groups = 1
            breakdown.agreement_boost = 0.0
            return 1.0
        
        # Count unique signal types (independent groups)
        signal_types = set(s.signal_type for s in signals)
        breakdown.independent_signal_groups = len(signal_types)
        
        # Calculate boost
        agreement = 1.0 + min(
            (len(signal_types) - 1) * self.AGREEMENT_BOOST_PER_GROUP,
            self.MAX_AGREEMENT_BOOST
        )
        
        breakdown.agreement_boost = agreement - 1.0
        
        self.logger.debug(
            "Signal agreement: groups=%d, agreement=%.4f",
            len(signal_types), agreement
        )
        
        return agreement
    
    def _calculate_data_freshness(
        self,
        signals: List[SignalEvidence],
        breakdown: ConfidenceBreakdown,
    ) -> float:
        """
        Calculate data freshness penalty.
        
        Older data reduces confidence as it may no longer be accurate.
        
        Freshness = 1.0 - min(hours_since_update × 0.02, 0.3)
        Max penalty = 0.3 (when data is 15+ hours old)
        """
        if not signals:
            breakdown.max_data_age_hours = 0.0
            breakdown.freshness_penalty = 0.0
            return 1.0
        
        # Calculate age of each signal's data
        now = datetime.utcnow()
        max_age = 0.0
        
        for signal in signals:
            if signal.data_timestamp:
                try:
                    dt_str = signal.data_timestamp
                    # Handle various ISO formats robustly
                    # Remove Z suffix if present
                    if dt_str.endswith("Z"):
                        dt_str = dt_str[:-1]
                    
                    # Try parsing with fromisoformat (handles most formats in Python 3.7+)
                    try:
                        dt = datetime.fromisoformat(dt_str)
                    except ValueError:
                        # Fallback: try parsing with timezone by replacing +HH:MM with +HHMM
                        import re
                        tz_match = re.search(r'([+-])(\d{2}):(\d{2})$', dt_str)
                        if tz_match:
                            dt_str = dt_str[:tz_match.start()] + tz_match.group(1) + tz_match.group(2) + tz_match.group(3)
                            dt = datetime.fromisoformat(dt_str)
                        else:
                            raise
                    
                    age_hours = (now - dt).total_seconds() / 3600.0
                    max_age = max(max_age, age_hours)
                except (ValueError, TypeError) as e:
                    self.logger.debug("Failed to parse timestamp '%s': %s", signal.data_timestamp, e)
                    pass
        
        breakdown.max_data_age_hours = max_age
        
        # Calculate freshness
        freshness = 1.0 - min(
            max_age * self.FRESHNESS_PENALTY_PER_HOUR,
            self.MAX_FRESHNESS_PENALTY
        )
        
        breakdown.freshness_penalty = 1.0 - freshness
        
        self.logger.debug(
            "Data freshness: max_age=%.2fh, freshness=%.4f",
            max_age, freshness
        )
        
        return freshness
    
    def _calculate_conflict_penalty(
        self,
        signals: List[SignalEvidence],
        conflict_groups: List[List[str]],
        breakdown: ConfidenceBreakdown,
    ) -> float:
        """
        Calculate penalty for conflicting evidence.
        
        When evidence conflicts (e.g., one signal says index is missing,
        another says it exists), confidence decreases.
        
        Conflict_Penalty = 1.0 - (conflict_count × 0.15)
        Min penalty = 0.4 (when 4+ conflicts detected)
        """
        if not conflict_groups:
            breakdown.conflict_count = 0
            breakdown.conflict_penalty_value = 0.0
            return 1.0
        
        # Count conflicts
        breakdown.conflict_count = len(conflict_groups)
        
        # Calculate penalty
        conflict_penalty = 1.0 - (
            len(conflict_groups) * self.CONFLICT_PENALTY_PER_CONFLICT
        )
        conflict_penalty = max(self.MIN_CONFLICT_PENALTY, conflict_penalty)
        
        breakdown.conflict_penalty_value = 1.0 - conflict_penalty
        
        self.logger.debug(
            "Conflict penalty: conflicts=%d, penalty=%.4f",
            len(conflict_groups), conflict_penalty
        )
        
        return conflict_penalty
    
    def explain_score(
        self,
        score: float,
        breakdown: ConfidenceBreakdown,
    ) -> Dict[str, Any]:
        """
        Generate detailed explanation of the confidence score.
        
        Args:
            score: The calculated confidence score
            breakdown: The detailed breakdown
            
        Returns:
            Dictionary with explanation and recommendations
        """
        explanation = {
            "score": round(score, 4),
            "rating": self._get_score_rating(score),
            "breakdown": breakdown.to_dict(),
            "human_readable": breakdown.to_human_readable(),
            "interpretation": self._interpret_score(score),
            "recommendations": [],
        }
        
        # Add recommendations based on breakdown
        if breakdown.completeness_penalty > 0.2:
            explanation["recommendations"].append(
                f"Collect missing evidence ({breakdown.missing_evidence_count} items) to improve confidence"
            )
        if breakdown.freshness_penalty > 0.1:
            explanation["recommendations"].append(
                f"Data is {breakdown.max_data_age_hours:.1f}h old - consider refreshing metrics"
            )
        if breakdown.conflict_penalty_value > 0.1:
            explanation["recommendations"].append(
                f"Resolve {breakdown.conflict_count} conflicting evidence items"
            )
        if breakdown.min_confidence < 0.7:
            explanation["recommendations"].append(
                f"Low confidence signal detected ({breakdown.min_confidence:.2f}) - investigate signal quality"
            )
        
        return explanation
    
    def _get_score_rating(self, score: float) -> str:
        """Get human-readable rating for the score."""
        if score >= 0.9:
            return "Very High"
        elif score >= 0.75:
            return "High"
        elif score >= 0.6:
            return "Moderate"
        elif score >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def _interpret_score(self, score: float) -> str:
        """Generate interpretation of the score."""
        if score >= 0.9:
            return ("Very high confidence. Multiple strong, independent signals "
                    "agree. Recommended for production decisions.")
        elif score >= 0.75:
            return ("High confidence. Strong evidence from multiple sources. "
                    "Safe for most operational decisions.")
        elif score >= 0.6:
            return ("Moderate confidence. Some evidence may be incomplete or stale. "
                    "Consider additional verification before critical changes.")
        elif score >= 0.4:
            return ("Low confidence. Evidence is incomplete or conflicting. "
                    "Further investigation recommended before action.")
        else:
            return ("Very low confidence. Insufficient or conflicting evidence. "
                    "Do not rely on this assessment for decisions.")
    
    def validate_inputs(
        self,
        signals: List[SignalEvidence],
        expected_evidence: List[ExpectedEvidence],
    ) -> Tuple[bool, List[str]]:
        """
        Validate inputs before calculating confidence.
        
        Args:
            signals: List of signals
            expected_evidence: List of expected evidence
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Validate signals
        for i, signal in enumerate(signals):
            if not signal.signal_id:
                errors.append(f"Signal {i}: Missing signal_id")
            if not signal.signal_type:
                errors.append(f"Signal {i}: Missing signal_type")
            if not 0.0 <= signal.confidence <= 1.0:
                errors.append(f"Signal {i}: Confidence {signal.confidence} out of range [0, 1]")
        
        # Validate expected evidence
        for i, evidence in enumerate(expected_evidence):
            if not evidence.evidence_type:
                errors.append(f"Expected evidence {i}: Missing evidence_type")
        
        return len(errors) == 0, errors


# ---------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------

def calculate_incident_confidence(
    signal_confidences: List[float],
    evidence_completeness: Dict[str, bool],
    signal_agreement_groups: Dict[str, int],
    data_freshness_hours: Dict[str, float],
    conflicting_evidence: List[Tuple[str, str]],
    seed: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Convenience function for calculating incident confidence.
    
    Args:
        signal_confidences: List of individual signal confidences
        evidence_completeness: Dict of evidence_type -> is_present
        signal_agreement_groups: Dict of group_name -> count
        data_freshness_hours: Dict of data_source -> age_in_hours
        conflicting_evidence: List of (signal1, signal2) tuples that conflict
        seed: Optional random seed for determinism
        
    Returns:
        Tuple of (confidence_score, explanation_dict)
    """
    scorer = ConfidenceScorer(seed=seed)
    
    # Build signals
    signals = [
        SignalEvidence(
            signal_id=f"sig_{i}",
            signal_name=f"signal_{i}",
            signal_type=f"type_{i % 4}",  # 4 types for agreement calculation
            confidence=conf,
        )
        for i, conf in enumerate(signal_confidences)
    ]
    
    # Build expected evidence
    expected = [
        ExpectedEvidence(evidence_type=k, is_required=v)
        for k, v in evidence_completeness.items()
    ]
    
    # Build conflict groups
    conflicts = []
    for s1, s2 in conflicting_evidence:
        conflicts.append([s1, s2])
    
    score, breakdown = scorer.calculate_confidence(
        signals=signals,
        expected_evidence=expected,
        conflict_groups=conflicts,
    )
    
    explanation = scorer.explain_score(score, breakdown)
    
    return score, explanation


# ---------------------------------------------------------------------
# Main execution for testing
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("Confidence Scoring Model - Example Usage")
    print("=" * 70)
    
    # Create scorer with seed for determinism
    scorer = ConfidenceScorer(seed=42)
    
    # Example: Slow query due to missing index
    signals = [
        SignalEvidence(
            signal_id="sig_001",
            signal_name="sequential_scan_detected",
            signal_type="query_metrics",
            confidence=0.90,
            evidence_ids=["ev_001", "ev_002"],
        ),
        SignalEvidence(
            signal_id="sig_002",
            signal_name="high_query_latency",
            signal_type="query_metrics",
            confidence=0.92,
            evidence_ids=["ev_003"],
        ),
        SignalEvidence(
            signal_id="sig_003",
            signal_name="missing_index_large_table",
            signal_type="table_stats",
            confidence=0.88,
            evidence_ids=["ev_004"],
        ),
        SignalEvidence(
            signal_id="sig_004",
            signal_name="row_estimation_error",
            signal_type="query_metrics",
            confidence=0.85,
            evidence_ids=["ev_005"],
        ),
    ]
    
    expected_evidence = [
        ExpectedEvidence(evidence_type="query_metrics", is_required=True),
        ExpectedEvidence(evidence_type="table_stats", is_required=True),
        ExpectedEvidence(evidence_type="index_health", is_required=False),
        ExpectedEvidence(evidence_type="configuration", is_required=False),
    ]
    
    score, breakdown = scorer.calculate_confidence(
        signals=signals,
        expected_evidence=expected_evidence,
    )
    
    print(f"\nConfidence Score: {score:.4f}")
    print("\n" + "-" * 70)
    print("Detailed Breakdown:")
    print("-" * 70)
    print(breakdown.to_human_readable())
    
    print("\n" + "-" * 70)
    print("JSON Output:")
    print("-" * 70)
    explanation = scorer.explain_score(score, breakdown)
    import json
    print(json.dumps(explanation, indent=2))

