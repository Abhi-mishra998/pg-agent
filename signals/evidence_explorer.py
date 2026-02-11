#!/usr/bin/env python3
"""
Evidence Explorer - Main Module for Evidence Visualization

Provides the Evidence Explorer class that manages evidence cards,
drill-down navigation, contradiction handling, and missing data display.

Goal: Make a senior DBA say "This system is not guessing."
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from signals.evidence_types import (
    EvidenceSource,
    EvidenceType,
    DeviationSeverity,
    EvidenceStatus,
    EVIDENCE_TYPE_TO_SOURCES,
)
from signals.evidence_card import (
    EvidenceCard,
    EvidenceGroup,
    Contradiction,
    MissingEvidence,
    EvidenceCollection,
    EvidenceCardFactory,
    MetricValue,
    Baseline,
    Deviation,
    EvidenceProvenance,
)


# =====================================================================
# EVIDENCE EXPLORER CLASS
# =====================================================================

class EvidenceExplorer:
    """
    Main Evidence Explorer class.
    
    Responsibilities:
    - Manage evidence collection and grouping
    - Handle drill-down navigation
    - Detect and display contradictions
    - Track and display missing evidence
    - Generate evidence summaries
    """
    
    def __init__(self, incident_id: str = "", log_level: str = "INFO"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        
        self.incident_id = incident_id
        self.collection = EvidenceCollection(incident_id=incident_id)
        
        # Navigation state
        self._selected_evidence_id: Optional[str] = None
        self._expanded_evidence_ids: List[str] = []
        self._collapsed_group_types: List[str] = []
        
        # Filter state
        self._filters: Dict[str, Any] = {
            "search": "",
            "evidence_type": "all",
            "source": "all",
            "status": "all",
            "severity": "all",
        }
    
    # =====================================================================
    # EVIDENCE MANAGEMENT
    # =====================================================================
    
    def add_evidence(self, card: EvidenceCard) -> None:
        """Add an evidence card to the collection."""
        self.collection.evidence.append(card)
        self.logger.debug(f"Added evidence: {card.evidence_id} - {card.title}")
    
    def add_evidence_from_metric(
        self,
        metric_name: str,
        current_value: float,
        evidence_type: EvidenceType,
        source: EvidenceSource,
        baseline_value: Optional[float] = None,
        unit: str = "",
        description: str = "",
        confidence: float = 0.5,
        raw_value: Optional[Dict[str, Any]] = None,
    ) -> EvidenceCard:
        """Add evidence from a metric measurement."""
        card = EvidenceCard.from_metric(
            metric_name=metric_name,
            current_value=current_value,
            evidence_type=evidence_type,
            source=source,
            baseline_value=baseline_value,
            unit=unit,
            description=description,
            confidence=confidence,
            raw_value=raw_value,
        )
        self.add_evidence(card)
        return card
    
    def add_pg_stat_statements_evidence(
        self,
        query_hash: str,
        mean_time_ms: float,
        calls: int,
        query_text: str = "",
        baseline_ms: float = 100.0,
    ) -> EvidenceCard:
        """Add evidence from pg_stat_statements."""
        data = {
            "query_hash": query_hash,
            "mean_time_ms": mean_time_ms,
            "calls": calls,
            "query": query_text,
        }
        
        return self.add_evidence_from_metric(
            metric_name="query_latency_ms",
            current_value=mean_time_ms,
            evidence_type=EvidenceType.QUERY,
            source=EvidenceSource.PG_STAT_STATEMENTS,
            baseline_value=baseline_ms,
            unit="ms",
            description=f"Query {query_hash[:8]}: Mean latency",
            confidence=0.9,
            raw_value=data,
        )
    
    def add_table_stats_evidence(
        self,
        table_name: str,
        seq_scans: int,
        idx_scans: int,
        row_count: int,
        last_analyze_days: int = 0,
    ) -> EvidenceCard:
        """Add evidence from table statistics."""
        total_scans = seq_scans + idx_scans
        seq_ratio = (seq_scans / total_scans * 100) if total_scans > 0 else 0
        
        # Determine confidence based on anomaly
        is_anomaly = seq_ratio > 90
        confidence = 0.95 if is_anomaly else 0.5
        
        data = {
            "table": table_name,
            "seq_scans": seq_scans,
            "idx_scans": idx_scans,
            "seq_scan_ratio": seq_ratio,
            "row_count": row_count,
            "last_analyze_days": last_analyze_days,
        }
        
        return self.add_evidence_from_metric(
            metric_name="seq_scan_ratio",
            current_value=seq_ratio,
            evidence_type=EvidenceType.TABLE_STATS,
            source=EvidenceSource.PG_STAT_USER_TABLES,
            baseline_value=10,  # 10% is normal
            unit="%",
            description=f"Table {table_name}: Sequential scan ratio",
            confidence=confidence,
            raw_value=data,
        )
    
    def add_lock_evidence(
        self,
        blocked_pid: int,
        blocking_pid: int,
        wait_duration_seconds: float,
        lock_type: str = "",
        blocked_query: str = "",
        blocking_query: str = "",
    ) -> EvidenceCard:
        """Add evidence for a blocking lock situation."""
        card = EvidenceCard(
            evidence_id=f"ev_lock_{blocked_pid}_{blocking_pid}",
            evidence_type=EvidenceType.LOCK,
            title=f"Blocking: PID {blocked_pid} ← {blocking_pid}",
            description=f"Process {blocked_pid} blocked by {blocking_pid} for {wait_duration_seconds}s",
            metric=MetricValue(
                name="wait_duration_seconds",
                value=wait_duration_seconds,
                unit="seconds",
                threshold=60,
                is_anomaly=wait_duration_seconds > 60,
            ),
            confidence=0.95,
            status=EvidenceStatus.CONFIRMED,
            provenance=EvidenceProvenance.from_source(
                EvidenceSource.PG_LOCKS,
                datetime.utcnow().isoformat()
            ),
            context={
                "blocked_pid": blocked_pid,
                "blocking_pid": blocking_pid,
                "lock_type": lock_type,
                "blocked_query": blocked_query,
                "blocking_query": blocking_query,
            },
            raw_value={
                "blocked_pid": blocked_pid,
                "blocking_pid": blocking_pid,
                "wait_duration": wait_duration_seconds,
                "lock_type": lock_type,
            },
        )
        self.add_evidence(card)
        return card
    
    def add_config_evidence(
        self,
        param_name: str,
        current_value: Any,
        recommended_value: Any,
        unit: str = "",
        context: str = "",
    ) -> EvidenceCard:
        """Add evidence for a configuration parameter."""
        card = EvidenceCard(
            evidence_id=f"ev_config_{param_name}",
            evidence_type=EvidenceType.CONFIG,
            title=f"Config: {param_name}",
            description=f"Configuration parameter {param_name}" + (f" - {context}" if context else ""),
            metric=MetricValue(
                name=param_name,
                value=current_value,
                unit=unit,
            ),
            baseline=Baseline(value=recommended_value, source="recommended"),
            deviation=Deviation.calculate(current_value, recommended_value),
            confidence=0.85,
            status=EvidenceStatus.CONFIRMED,
            provenance=EvidenceProvenance.from_source(
                EvidenceSource.PG_SETTINGS,
                datetime.utcnow().isoformat()
            ),
            context={"current": current_value, "recommended": recommended_value},
        )
        self.add_evidence(card)
        return card
    
    def add_missing_evidence(
        self,
        evidence_type: str,
        description: str,
        required_for: List[str],
        confidence_penalty: float,
        collection_action: Optional[str] = None,
        collection_risk: str = "low",
        is_required: bool = True,
    ) -> MissingEvidence:
        """Record missing evidence that could improve confidence."""
        missing = MissingEvidence(
            evidence_type=evidence_type,
            description=description,
            required_for=required_for,
            confidence_penalty=confidence_penalty,
            collection_action=collection_action,
            collection_risk=collection_risk,
            is_required=is_required,
        )
        self.collection.missing_evidence.append(missing)
        return missing
    
    # =====================================================================
    # EVIDENCE ANALYSIS
    # =====================================================================
    
    def analyze_contradictions(self) -> List[Contradiction]:
        """
        Analyze evidence for contradictions.
        
        Returns list of detected contradictions with supporting and
        contradicting evidence.
        """
        contradictions: List[Contradiction] = []
        
        # Example contradiction patterns
        # Pattern 1: Index scan reported but no index on filter column
        index_scan_cards = [c for c in self.collection.evidence 
                           if c.evidence_type == EvidenceType.INDEX_STATS]
        seq_scan_cards = [c for c in self.collection.evidence 
                         if c.evidence_type == EvidenceType.TABLE_STATS]
        
        # Check for sequential scan vs index scan contradiction
        for seq_card in seq_scan_cards:
            if seq_card.deviation and seq_card.deviation.percent > 90:
                # Check if there's supporting index evidence
                has_index_evidence = any(
                    c.context.get("index_columns", []) == [] and 
                    c.title.lower().find("missing") >= 0
                    for c in self.collection.evidence
                )
                
                if not has_index_evidence:
                    # Create contradiction record
                    contra = Contradiction(
                        claim="Sequential scan occurring without index",
                        supporting_evidence=[seq_card.evidence_id],
                        contradicting_evidence=[],
                        explanation="Sequential scan detected but no evidence of index on filter column. "
                                   "This confirms the sequential scan issue.",
                        resolution_status="resolved",
                        uncertainty_impact=0.0,
                    )
                    contradictions.append(contra)
        
        self.collection.contradictions = contradictions
        return contradictions
    
    def calculate_confidence_breakdown(self) -> Dict[str, float]:
        """Calculate confidence score breakdown by component."""
        evidence_count = len(self.collection.evidence)
        
        # Base confidence from evidence
        if evidence_count > 0:
            base_confidence = sum(e.confidence for e in self.collection.evidence) / evidence_count
        else:
            base_confidence = 0.5
        
        # Completeness penalty for missing evidence
        missing_penalty = len(self.collection.missing_evidence) * 0.05
        completeness = max(0.5, 1.0 - missing_penalty)
        
        # Contradiction penalty
        contradiction_penalty = sum(
            c.uncertainty_impact for c in self.collection.contradictions
        )
        conflict = max(0.4, 1.0 - contradiction_penalty)
        
        # Data freshness penalty
        stale_count = sum(
            1 for e in self.collection.evidence
            if e.provenance and e.provenance.freshness_status == "stale"
        )
        freshness = max(0.7, 1.0 - (stale_count * 0.05))
        
        # Calculate overall
        overall = base_confidence * completeness * conflict * freshness
        
        breakdown = {
            "base_confidence": round(base_confidence, 3),
            "evidence_completeness": round(completeness, 3),
            "signal_agreement": 1.0,  # Placeholder for agreement calculation
            "data_freshness": round(freshness, 3),
            "conflict_penalty": round(conflict, 3),
            "overall_confidence": round(overall, 3),
        }
        
        self.collection.overall_confidence = breakdown["overall_confidence"]
        self.collection.confidence_breakdown = breakdown
        
        return breakdown
    
    def group_evidence(self) -> List[EvidenceGroup]:
        """Group evidence by type for display."""
        groups = self.collection.group_evidence_by_type()
        self.collection.groups = groups
        return groups
    
    def get_source_summary(self) -> Dict[str, int]:
        """Get summary of evidence by source."""
        return self.collection.calculate_source_counts()
    
    # =====================================================================
    # NAVIGATION & FILTERS
    # =====================================================================
    
    def select_evidence(self, evidence_id: str) -> Optional[EvidenceCard]:
        """Select an evidence card for drill-down."""
        self._selected_evidence_id = evidence_id
        return self.collection.get_evidence_by_id(evidence_id)
    
    def toggle_evidence_expansion(self, evidence_id: str) -> None:
        """Toggle expansion state of an evidence card."""
        if evidence_id in self._expanded_evidence_ids:
            self._expanded_evidence_ids.remove(evidence_id)
        else:
            self._expanded_evidence_ids.append(evidence_id)
    
    def toggle_group_collapse(self, group_type: str) -> None:
        """Toggle collapse state of an evidence group."""
        if group_type in self._collapsed_group_types:
            self._collapsed_group_types.remove(group_type)
        else:
            self._collapsed_group_types.append(group_type)
    
    def set_filter(self, filter_type: str, value: Any) -> None:
        """Set a filter for evidence display."""
        if filter_type in self._filters:
            self._filters[filter_type] = value
    
    def get_filtered_evidence(self) -> List[EvidenceCard]:
        """Get evidence matching current filters."""
        filtered = self.collection.evidence
        
        # Search filter
        if self._filters["search"]:
            search = self._filters["search"].lower()
            filtered = [
                e for e in filtered
                if search in e.title.lower() or search in e.description.lower()
            ]
        
        # Evidence type filter
        if self._filters["evidence_type"] != "all":
            filter_type = self._filters["evidence_type"]
            filtered = [
                e for e in filtered
                if e.evidence_type.value == filter_type
            ]
        
        # Source filter
        if self._filters["source"] != "all":
            source_filter = self._filters["source"]
            filtered = [
                e for e in filtered
                if e.provenance and e.provenance.source_name == source_filter
            ]
        
        # Status filter
        if self._filters["status"] != "all":
            status_filter = self._filters["status"]
            filtered = [
                e for e in filtered
                if e.status.value == status_filter
            ]
        
        # Severity filter
        if self._filters["severity"] != "all":
            severity_filter = self._filters["severity"]
            filtered = [
                e for e in filtered
                if e.deviation and e.deviation.severity.value == severity_filter
            ]
        
        return filtered
    
    # =====================================================================
    # RENDERING OUTPUT
    # =====================================================================
    
    def render_terminal(self, include_details: bool = True) -> str:
        """Render evidence for terminal output."""
        lines = []
        lines.append("=" * 80)
        lines.append("📊 EVIDENCE EXPLORER")
        lines.append(f"Incident: {self.incident_id}")
        lines.append(f"Evidence Count: {len(self.collection.evidence)}")
        lines.append("=" * 80)
        
        # Confidence summary
        breakdown = self.calculate_confidence_breakdown()
        lines.append("")
        lines.append(f"🧠 CONFIDENCE: {breakdown['overall_confidence']:.0%}")
        lines.append(f"  Base: {breakdown['base_confidence']:.0%} | "
                    f"Completeness: {breakdown['evidence_completeness']:.0%} | "
                    f"Freshness: {breakdown['data_freshness']:.0%}")
        
        # Source summary
        source_counts = self.get_source_summary()
        lines.append("")
        lines.append("📍 EVIDENCE SOURCES:")
        for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {source}: {count}")
        
        # Evidence groups
        groups = self.group_evidence()
        lines.append("")
        lines.append("📋 EVIDENCE BY TYPE:")
        
        for group in groups:
            status_icon = "▼" if group.group_type not in self._collapsed_group_types else "▶"
            lines.append(f"  {status_icon} {group.icon} {group.display_name} ({group.count})")
            
            if group.group_type not in self._collapsed_group_types:
                for card in group.evidence:
                    lines.append(self._render_card_terminal(card, include_details))
        
        # Contradictions
        contradictions = self.analyze_contradictions()
        if contradictions:
            lines.append("")
            lines.append("⚠️ CONTRADICTIONS:")
            for contra in contradictions:
                lines.append(f"  Claim: {contra.claim}")
                lines.append(f"  Status: {contra.resolution_status}")
                if contra.explanation:
                    lines.append(f"  Explanation: {contra.explanation}")
        
        # Missing evidence
        if self.collection.missing_evidence:
            lines.append("")
            lines.append("❓ MISSING EVIDENCE:")
            for missing in self.collection.missing_evidence:
                lines.append(f"  - {missing.evidence_type}: {missing.description}")
                lines.append(f"    Confidence penalty: -{missing.confidence_penalty:.0%}")
        
        return "\n".join(lines)
    
    def _render_card_terminal(self, card: EvidenceCard, include_details: bool = True) -> str:
        """Render a single evidence card for terminal."""
        lines = []
        
        # Header
        confidence_badge = card.get_confidence_badge()
        lines.append(f"    {card.evidence_type.icon} {card.title}")
        lines.append(f"      Confidence: {confidence_badge['icon']} {confidence_badge['text']} ({confidence_badge['label']})")
        
        # Deviation
        if card.deviation:
            dev_display = card.get_deviation_display()
            lines.append(f"      Value: {dev_display['current']} → {dev_display['baseline']} ({dev_display['icon']} {dev_display['percent']})")
        
        # Source
        if card.provenance:
            lines.append(f"      Source: {card.provenance.icon} {card.provenance.source_name}")
        
        # Details
        if include_details and card.evidence_id in self._expanded_evidence_ids:
            if card.raw_value:
                lines.append(f"      Raw: {str(card.raw_value)[:100]}...")
        
        return "\n".join(lines)
    
    def render_markdown(self, include_details: bool = True) -> str:
        """Render evidence as Markdown for documentation."""
        lines = []
        
        lines.append("# 📊 Evidence Explorer Report")
        lines.append("")
        lines.append(f"**Incident ID:** {self.incident_id}")
        lines.append(f"**Generated:** {self.collection.generated_at}")
        lines.append(f"**Evidence Count:** {len(self.collection.evidence)}")
        lines.append("")
        
        # Confidence
        breakdown = self.calculate_confidence_breakdown()
        lines.append("## 🧠 Confidence Score")
        lines.append("")
        lines.append(f"**Overall Confidence:** {breakdown['overall_confidence']:.0%}")
        lines.append("")
        lines.append("| Component | Score |")
        lines.append("|-----------|-------|")
        for key, value in breakdown.items():
            if key != "overall_confidence":
                lines.append(f"| {key.replace('_', ' ').title()} | {value:.0%} |")
        lines.append("")
        
        # Source summary
        source_counts = self.get_source_summary()
        lines.append("## 📍 Evidence Sources")
        lines.append("")
        lines.append("| Source | Count |")
        lines.append("|--------|-------|")
        for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {source} | {count} |")
        lines.append("")
        
        # Evidence by type
        groups = self.group_evidence()
        lines.append("## 📋 Evidence by Type")
        lines.append("")
        
        for group in groups:
            lines.append(f"### {group.icon} {group.display_name}")
            lines.append("")
            
            for card in group.evidence:
                lines.append(f"#### {card.title}")
                lines.append("")
                lines.append(f"**Description:** {card.description}")
                lines.append(f"**Confidence:** {card.confidence:.0%}")
                
                if card.deviation:
                    dev = card.get_deviation_display()
                    lines.append(f"**Deviation:** {dev['percent']} ({dev['severity']})")
                
                if card.provenance:
                    lines.append(f"**Source:** {card.provenance.icon} {card.provenance.source_name}")
                
                lines.append("")
        
        # Contradictions
        contradictions = self.analyze_contradictions()
        if contradictions:
            lines.append("## ⚠️ Contradictions")
            lines.append("")
            for contra in contradictions:
                lines.append(f"### {contra.claim}")
                lines.append("")
                lines.append(f"**Status:** {contra.resolution_status}")
                lines.append(f"**Explanation:** {contra.explanation}")
                lines.append("")
        
        # Missing evidence
        if self.collection.missing_evidence:
            lines.append("## ❓ Missing Evidence")
            lines.append("")
            for missing in self.collection.missing_evidence:
                lines.append(f"### {missing.evidence_type}")
                lines.append("")
                lines.append(f"**Description:** {missing.description}")
                lines.append(f"**Confidence Penalty:** -{missing.confidence_penalty:.0%}")
                if missing.collection_action:
                    lines.append(f"**Collection Action:** `{missing.collection_action}`")
                lines.append("")
        
        return "\n".join(lines)
    
    def render_html(self) -> str:
        """Render evidence as HTML for web display."""
        # This would generate HTML for the Evidence Explorer UI
        # For now, return a placeholder
        return f"""
        <div class="evidence-explorer">
            <h1>Evidence Explorer</h1>
            <p>Incident: {self.incident_id}</p>
            <p>Evidence Count: {len(self.collection.evidence)}</p>
            <!-- Full HTML rendering would be implemented here -->
        </div>
        """
    
    def to_dict(self) -> Dict[str, Any]:
        """Export evidence collection as dictionary."""
        self.calculate_confidence_breakdown()
        self.group_evidence()
        
        return {
            "incident_id": self.incident_id,
            "collection": self.collection.to_dict(),
            "navigation": {
                "selected_evidence_id": self._selected_evidence_id,
                "expanded_evidence_ids": self._expanded_evidence_ids,
                "collapsed_group_types": self._collapsed_group_types,
            },
            "filters": self._filters,
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save evidence collection to JSON file."""
        import json
        
        data = self.to_dict()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Saved evidence collection to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str, incident_id: str = "") -> "EvidenceExplorer":
        """Load evidence collection from JSON file."""
        import json
        
        path = Path(filepath)
        with open(path, 'r') as f:
            data = json.load(f)
        
        explorer = cls(incident_id=incident_id or data.get("incident_id", ""))
        
        # Reconstruct evidence
        if "collection" in data:
            collection_data = data["collection"]
            explorer.collection = EvidenceCollection.from_dict(collection_data)
        
        # Restore navigation state
        if "navigation" in data:
            nav = data["navigation"]
            explorer._selected_evidence_id = nav.get("selected_evidence_id")
            explorer._expanded_evidence_ids = nav.get("expanded_evidence_ids", [])
            explorer._collapsed_group_types = nav.get("collapsed_group_types", [])
        
        # Restore filters
        if "filters" in data:
            explorer._filters.update(data["filters"])
        
        return explorer


# =====================================================================
# CONVENIENCE FUNCTIONS
# =====================================================================

def create_evidence_explorer(
    incident_id: str,
    evidence_data: List[Dict[str, Any]] = None,
) -> EvidenceExplorer:
    """
    Create an EvidenceExplorer instance with optional initial evidence.
    
    Args:
        incident_id: The incident ID this evidence is for
        evidence_data: Optional list of evidence dictionaries
        
    Returns:
        Configured EvidenceExplorer instance
    """
    explorer = EvidenceExplorer(incident_id=incident_id)
    
    if evidence_data:
        for data in evidence_data:
            card = EvidenceCard.from_dict(data)
            explorer.add_evidence(card)
    
    return explorer


def build_evidence_from_incident(incident_data: Dict[str, Any]) -> EvidenceExplorer:
    """
    Build an EvidenceExplorer from an incident output dictionary.
    
    Args:
        incident_data: Full incident output dictionary
        
    Returns:
        EvidenceExplorer with all evidence loaded
    """
    summary = incident_data.get("summary", {})
    incident_id = summary.get("incident_id", "")
    
    explorer = EvidenceExplorer(incident_id=incident_id)
    
    # Add evidence from incident
    for evidence_data in incident_data.get("evidence", []):
        card = EvidenceCard.from_dict(evidence_data)
        explorer.add_evidence(card)
    
    # Calculate and display confidence
    explorer.calculate_confidence_breakdown()
    explorer.group_evidence()
    
    return explorer


# =====================================================================
# EXPORTS
# =====================================================================

__all__ = [
    "EvidenceExplorer",
    "create_evidence_explorer",
    "build_evidence_from_incident",
]

