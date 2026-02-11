#!/usr/bin/env python3
"""
EvidenceBuilder - Evidence Collection Module

Builds structured evidence from SignalResult.
Designed for incident response, explainability, and LLM grounding.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from signals.signal_engine import SignalResult, Signal


# ---------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------

@dataclass
class Evidence:
    id: str
    source_signal: str
    type: str
    content: str
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_signals: List[str] = field(default_factory=list)


@dataclass
class EvidenceCollection:
    evidence: List[Evidence]
    overall_confidence: float
    signal_count: int
    evidence_types: Dict[str, int]
    analysis: Dict[str, Any]
    processing_time: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Evidence Builder
# ---------------------------------------------------------------------

class EvidenceBuilder:
    """
    Converts SignalResult into EvidenceCollection.

    Responsibilities:
    - Evidence extraction
    - Confidence aggregation
    - Evidence correlation
    - Preserve SignalEngine analysis
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        correlation_enabled: bool = True,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.min_confidence = min_confidence
        self.correlation_enabled = correlation_enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        signal_result: SignalResult,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvidenceCollection:
        """
        Build evidence from SignalResult.
        """

        # 🛑 HARD CONTRACT CHECK (THIS FIXES YOUR CRASHES)
        if not isinstance(signal_result, SignalResult):
            raise TypeError(
                f"EvidenceBuilder.build expects SignalResult, got {type(signal_result)}"
            )

        signals: List[Signal] = signal_result.signals
        context = context or {}

        self.logger.info("Building evidence from %d signals", len(signals))

        evidence_items: List[Evidence] = []

        for signal in signals:
            evidence_items.extend(
                self._extract_from_signal(signal, context)
            )

        # Confidence filtering
        filtered = [
            e for e in evidence_items if e.confidence >= self.min_confidence
        ]

        # Correlation
        if self.correlation_enabled and filtered:
            self._correlate(filtered)

        overall_confidence = self._aggregate_confidence(filtered)

        evidence_types: Dict[str, int] = {}
        for e in filtered:
            evidence_types[e.type] = evidence_types.get(e.type, 0) + 1

        return EvidenceCollection(
            evidence=filtered,
            overall_confidence=overall_confidence,
            signal_count=len(signals),
            evidence_types=evidence_types,
            analysis=signal_result.analysis,
            processing_time=signal_result.processing_time,
            metadata={
                "filtered_signals": signal_result.filtered_count,
                "min_confidence": self.min_confidence,
                "correlation_enabled": self.correlation_enabled,
            },
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_from_signal(
        self,
        signal: Signal,
        context: Dict[str, Any],
    ) -> List[Evidence]:
        """
        Convert a Signal into one or more Evidence objects.
        """
        evidence: List[Evidence] = []

        evidence.append(Evidence(
            id=f"ev_{signal.id}",
            source_signal=signal.id,
            type=signal.type,
            content=str(signal.data),
            confidence=signal.confidence,
            metadata={
                "signal_name": signal.name,
                "signal_type": signal.type,
                **signal.metadata,
            },
        ))

        # Optional explainability evidence
        explain = signal.metadata.get("explain")
        if explain:
            evidence.append(Evidence(
                id=f"ev_{signal.id}_explain",
                source_signal=signal.id,
                type="explanation",
                content=explain,
                confidence=min(signal.confidence + 0.05, 1.0),
            ))

        return evidence

    # ------------------------------------------------------------------
    # Correlation & Aggregation
    # ------------------------------------------------------------------

    def _correlate(self, evidence: List[Evidence]) -> None:
        ids = [e.source_signal for e in evidence]
        for e in evidence:
            e.related_signals = [i for i in ids if i != e.source_signal]

    def _aggregate_confidence(self, evidence: List[Evidence]) -> float:
        if not evidence:
            return 0.0
        return round(
            sum(e.confidence for e in evidence) / len(evidence),
            2
        )

    # ------------------------------------------------------------------
    # Reporting Helpers
    # ------------------------------------------------------------------

    def summary(self, collection: EvidenceCollection) -> Dict[str, Any]:
        return {
            "total_evidence": len(collection.evidence),
            "overall_confidence": collection.overall_confidence,
            "signal_count": collection.signal_count,
            "evidence_types": collection.evidence_types,
            "analysis": collection.analysis,
            "processing_time": collection.processing_time,
            "generated_at": collection.timestamp,
        }