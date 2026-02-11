#!/usr/bin/env python3
"""
IncidentRenderer - Human-Readable Incident Rendering Layer

Transforms structured incident data into clean, executive-friendly output.

Design Principles:
- Render ONLY from structured data (no hallucinations)
- No new facts added - all data sourced from incident object
- Professional, clean format suitable for Slack, email, terminal
- Visual formatting with emojis for quick scanning
- Hierarchical information architecture
- Clear confidence indicators and caveats
"""

import logging
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


# =========================================================================
# Data Models
# =========================================================================

class RenderFormat(Enum):
    """Output format options."""
    TERMINAL = "terminal"       # Rich ANSI color output
    MARKDOWN = "markdown"       # GitHub-flavored markdown
    SLACK = "slack"            # Slack message format
    EMAIL = "email"            # HTML email format
    JSON = "json"              # Structured JSON


@dataclass
class RenderContext:
    """Rendering context with configuration."""
    format: RenderFormat = RenderFormat.TERMINAL
    include_evidence_details: bool = True
    include_recommendations_details: bool = True
    include_metadata: bool = True
    max_evidence_items: int = 10
    max_recommendations: int = 5
    confidence_threshold_warning: float = 0.75
    colors_enabled: bool = True


# =========================================================================
# Terminal ANSI Colors
# =========================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    
    # Background
    BG_RED = "\033[101m"
    BG_YELLOW = "\033[103m"
    
    @staticmethod
    def severity_color(severity: str) -> str:
        """Get color for severity level."""
        if "CRITICAL" in severity:
            return Colors.RED
        elif "HIGH" in severity:
            return Colors.YELLOW
        elif "MEDIUM" in severity:
            return Colors.CYAN
        return Colors.GREEN
    
    @staticmethod
    def confidence_color(confidence: float) -> str:
        """Get color for confidence score."""
        if confidence >= 0.85:
            return Colors.GREEN
        elif confidence >= 0.75:
            return Colors.CYAN
        elif confidence >= 0.60:
            return Colors.YELLOW
        return Colors.RED


# =========================================================================
# Confidence and Evidence Formatters
# =========================================================================

class ConfidenceFormatter:
    """Formats confidence scores for human readability."""
    
    @staticmethod
    def format_confidence(confidence: float) -> str:
        """Format confidence as percentage with rating."""
        percentage = confidence * 100
        
        if confidence >= 0.90:
            rating = "Very High"
            emoji = "✅"
        elif confidence >= 0.80:
            rating = "High"
            emoji = "✅"
        elif confidence >= 0.70:
            rating = "Moderate"
            emoji = "⚠️"
        elif confidence >= 0.60:
            rating = "Low"
            emoji = "⚠️"
        else:
            rating = "Very Low"
            emoji = "❌"
        
        return f"{emoji} {percentage:.0f}% ({rating})"
    
    @staticmethod
    def format_confidence_with_explanation(
        confidence: float,
        breakdown: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format confidence with detailed explanation."""
        text = ConfidenceFormatter.format_confidence(confidence)
        
        if breakdown:
            components = breakdown.get("components", {})
            if components:
                text += "\n  Components:"
                for key, value in components.items():
                    if isinstance(value, (int, float)):
                        text += f"\n    - {key.replace('_', ' ').title()}: {value:.2%}"
        
        return text


class EvidenceFormatter:
    """Formats evidence items for display."""
    
    @staticmethod
    def format_evidence_item(evidence: Dict[str, Any]) -> str:
        """Format a single evidence item."""
        evidence_type = evidence.get("evidence_type", "unknown").replace("_", " ").title()
        value = evidence.get("value", "N/A")
        source = evidence.get("source", "system")
        confidence = evidence.get("confidence", 0.0)
        
        confidence_bar = EvidenceFormatter._make_confidence_bar(confidence)
        
        return (
            f"  • {evidence_type}\n"
            f"    Value: {value}\n"
            f"    Source: {source} | Confidence: {confidence_bar}"
        )
    
    @staticmethod
    def format_evidence_list(
        evidence_items: List[Dict[str, Any]],
        max_items: Optional[int] = None
    ) -> str:
        """Format multiple evidence items."""
        if not evidence_items:
            return "  (No evidence collected)"
        
        items = evidence_items[:max_items] if max_items else evidence_items
        formatted = "\n".join(
            EvidenceFormatter.format_evidence_item(item)
            for item in items
        )
        
        if max_items and len(evidence_items) > max_items:
            formatted += f"\n  ... and {len(evidence_items) - max_items} more items"
        
        return formatted
    
    @staticmethod
    def _make_confidence_bar(confidence: float, width: int = 10) -> str:
        """Create a visual confidence bar."""
        filled = int(confidence * width)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}] {confidence:.0%}"


# =========================================================================
# Incident Renderer
# =========================================================================

class IncidentRenderer:
    """
    Renders structured incident data into human-readable formats.
    
    Main entry point for generating incident summaries in various formats.
    """
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
    
    def render(
        self,
        incident_data: Dict[str, Any],
        context: Optional[RenderContext] = None,
    ) -> str:
        """
        Render incident data to human-readable format.
        
        Args:
            incident_data: Structured incident dictionary
            context: Rendering context (format, options, etc.)
            
        Returns:
            Formatted incident report as string
        """
        context = context or RenderContext()
        
        # Route to appropriate formatter
        if context.format == RenderFormat.TERMINAL:
            return self._render_terminal(incident_data, context)
        elif context.format == RenderFormat.MARKDOWN:
            return self._render_markdown(incident_data, context)
        elif context.format == RenderFormat.SLACK:
            return self._render_slack(incident_data, context)
        elif context.format == RenderFormat.EMAIL:
            return self._render_email(incident_data, context)
        elif context.format == RenderFormat.JSON:
            return json.dumps(incident_data, indent=2)
        else:
            return self._render_terminal(incident_data, context)
    
    # =====================================================================
    # Terminal/Console Rendering
    # =====================================================================
    
    def _render_terminal(
        self,
        incident_data: Dict[str, Any],
        context: RenderContext,
    ) -> str:
        """Render for terminal with ANSI colors."""
        lines = []
        
        # Header
        lines.append(self._render_header_terminal(incident_data, context))
        
        # Summary section
        lines.append(self._render_summary_terminal(incident_data, context))
        
        # Root cause section
        lines.append(self._render_root_cause_terminal(incident_data, context))
        
        # Evidence section
        if context.include_evidence_details:
            lines.append(self._render_evidence_terminal(incident_data, context))
        
        # Impact section
        lines.append(self._render_impact_terminal(incident_data, context))
        
        # Recommendations section
        if context.include_recommendations_details:
            lines.append(self._render_recommendations_terminal(incident_data, context))
        
        # Confidence section
        lines.append(self._render_confidence_terminal(incident_data, context))
        
        # Footer/metadata
        if context.include_metadata:
            lines.append(self._render_metadata_terminal(incident_data, context))
        
        return "\n".join(lines)
    
    def _render_header_terminal(
        self,
        incident_data: Dict[str, Any],
        context: RenderContext,
    ) -> str:
        """Render incident header."""
        lines = []
        
        # Top separator
        lines.append("=" * 80)
        
        # Title with severity emoji
        severity = incident_data.get("severity", "UNKNOWN")
        severity_emoji = self._get_severity_emoji(severity)
        incident_id = incident_data.get("incident_id", "unknown")
        
        title_line = f"{severity_emoji} INCIDENT SUMMARY - {incident_id}"
        lines.append(title_line)
        
        # Severity line
        lines.append("-" * 80)
        
        return "\n".join(lines)
    
    def _render_summary_terminal(
        self,
        incident_data: Dict[str, Any],
        context: RenderContext,
    ) -> str:
        """Render incident summary."""
        lines = []
        lines.append("")
        lines.append("📋 SUMMARY")
        lines.append("-" * 80)
        
        # Incident type
        incident_type = incident_data.get("category", "UNKNOWN").replace("_", " ")
        lines.append(f"  Type: {incident_type}")
        
        # Status
        status = incident_data.get("status", "UNKNOWN")
        status_emoji = self._get_status_emoji(status)
        lines.append(f"  Status: {status_emoji} {status}")
        
        # Severity
        severity = incident_data.get("severity", "UNKNOWN")
        severity_color = Colors.severity_color(severity)
        if context.colors_enabled:
            severity_text = f"{severity_color}{severity}{Colors.RESET}"
        else:
            severity_text = severity
        lines.append(f"  Severity: {severity_text}")
        
        # Timestamp
        timestamp = incident_data.get("timestamp", datetime.utcnow().isoformat())
        lines.append(f"  Detected: {timestamp}")
        
        return "\n".join(lines)
    
    def _render_root_cause_terminal(
        self,
        incident_data: Dict[str, Any],
        context: RenderContext,
    ) -> str:
        """Render root cause analysis."""
        lines = []
        lines.append("")
        lines.append("🔍 ROOT CAUSE")
        lines.append("-" * 80)
        
        root_causes = incident_data.get("root_causes", [])
        
        if not root_causes:
            lines.append("  (Root cause analysis pending)")
            return "\n".join(lines)
        
        for idx, cause in enumerate(root_causes[:3], 1):  # Top 3 root causes
            cause_type = cause.get("type", "UNKNOWN").replace("_", " ")
            description = cause.get("description", "No description")
            confidence = cause.get("confidence", 0.0)
            
            lines.append(f"  {idx}. {cause_type}")
            lines.append(f"     Description: {description}")
            
            confidence_color = Colors.confidence_color(confidence)
            if context.colors_enabled:
                conf_text = f"{confidence_color}{confidence:.0%}{Colors.RESET}"
            else:
                conf_text = f"{confidence:.0%}"
            lines.append(f"     Confidence: {conf_text}")
        
        return "\n".join(lines)
    
    def _render_evidence_terminal(
        self,
        incident_data: Dict[str, Any],
        context: RenderContext,
    ) -> str:
        """Render supporting evidence."""
        lines = []
        lines.append("")
        lines.append("📊 EVIDENCE")
        lines.append("-" * 80)
        
        evidence_list = incident_data.get("evidence", [])
        
        if not evidence_list:
            lines.append("  (No evidence collected)")
            return "\n".join(lines)
        
        formatted_evidence = EvidenceFormatter.format_evidence_list(
            evidence_list,
            max_items=context.max_evidence_items
        )
        lines.append(formatted_evidence)
        
        return "\n".join(lines)
    
    def _render_impact_terminal(
        self,
        incident_data: Dict[str, Any],
        context: RenderContext,
    ) -> str:
        """Render business impact."""
        lines = []
        lines.append("")
        lines.append("💥 IMPACT")
        lines.append("-" * 80)
        
        impact = incident_data.get("impact", {})
        
        if not impact:
            lines.append("  (Impact assessment pending)")
            return "\n".join(lines)
        
        # Affected queries
        affected_queries = impact.get("affected_queries", 0)
        if affected_queries:
            lines.append(f"  Affected Queries: {affected_queries}")
        
        # Performance degradation
        degradation = impact.get("performance_degradation", "")
        if degradation:
            lines.append(f"  Performance: {degradation}")
        
        # Estimated users/applications impacted
        estimated_impact = impact.get("estimated_impact", "")
        if estimated_impact:
            lines.append(f"  Estimated Impact: {estimated_impact}")
        
        # Business context
        business_context = impact.get("business_context", "")
        if business_context:
            lines.append(f"  Context: {business_context}")
        
        return "\n".join(lines)
    
    def _render_recommendations_terminal(
        self,
        incident_data: Dict[str, Any],
        context: RenderContext,
    ) -> str:
        """Render recommended actions."""
        lines = []
        lines.append("")
        lines.append("🛠️ RECOMMENDED ACTIONS")
        lines.append("-" * 80)
        
        recommendations = incident_data.get("recommendations", [])
        
        if not recommendations:
            lines.append("  (No recommendations available)")
            return "\n".join(lines)
        
        for idx, rec in enumerate(recommendations[:context.max_recommendations], 1):
            action = rec.get("action", "Unknown action")
            risk = rec.get("risk", "UNKNOWN")
            risk_emoji = self._get_risk_emoji(risk)
            
            lines.append(f"  {idx}. {action}")
            lines.append(f"     Risk Level: {risk_emoji} {risk}")
            
            # Add details if available
            if rec.get("estimated_impact"):
                lines.append(f"     Expected Impact: {rec['estimated_impact']}")
            if rec.get("estimated_time"):
                lines.append(f"     Estimated Time: {rec['estimated_time']}")
        
        if len(recommendations) > context.max_recommendations:
            lines.append(f"\n  ... and {len(recommendations) - context.max_recommendations} more recommendations")
        
        return "\n".join(lines)
    
    def _render_confidence_terminal(
        self,
        incident_data: Dict[str, Any],
        context: RenderContext,
    ) -> str:
        """Render confidence assessment."""
        lines = []
        lines.append("")
        lines.append("🧠 CONFIDENCE")
        lines.append("-" * 80)
        
        confidence_score = incident_data.get("confidence_score", 0.0)
        confidence_text = ConfidenceFormatter.format_confidence(confidence_score)
        lines.append(f"  {confidence_text}")
        
        # Show warnings if below threshold
        if confidence_score < context.confidence_threshold_warning:
            lines.append("")
            lines.append(f"  ⚠️  Confidence below {context.confidence_threshold_warning:.0%} threshold")
            lines.append("  Additional clarification questions may improve accuracy:")
            
            clarification_qs = incident_data.get("clarification_questions", [])
            if clarification_qs:
                for idx, q in enumerate(clarification_qs[:3], 1):
                    q_text = q.get("question_text", "?")
                    lines.append(f"    {idx}. {q_text}")
        
        # Breakdown if available
        if incident_data.get("confidence_breakdown"):
            breakdown = incident_data["confidence_breakdown"]
            lines.append("")
            lines.append("  Components:")
            for key, value in breakdown.items():
                if isinstance(value, (int, float)):
                    lines.append(f"    - {key.replace('_', ' ').title()}: {value:.2%}")
        
        return "\n".join(lines)
    
    def _render_metadata_terminal(
        self,
        incident_data: Dict[str, Any],
        context: RenderContext,
    ) -> str:
        """Render metadata footer."""
        lines = []
        lines.append("")
        lines.append("=" * 80)
        
        # Generated at
        generated_at = incident_data.get("generated_at", datetime.utcnow().isoformat())
        lines.append(f"Generated: {generated_at}")
        
        # Agent version
        agent_version = incident_data.get("agent_version", "unknown")
        lines.append(f"Agent: pg-agent v{agent_version}")
        
        # Caveats
        caveats = incident_data.get("caveats", [])
        if caveats:
            lines.append("\nCaveats:")
            for caveat in caveats[:3]:
                lines.append(f"  - {caveat}")
        
        return "\n".join(lines)
    
    # =====================================================================
    # Markdown Rendering
    # =====================================================================
    
    def _render_markdown(
        self,
        incident_data: Dict[str, Any],
        context: RenderContext,
    ) -> str:
        """Render for Markdown (GitHub-flavored)."""
        lines = []
        
        # Header
        incident_id = incident_data.get("incident_id", "unknown")
        lines.append(f"# 🚨 Incident Report: {incident_id}")
        lines.append("")
        
        # Summary
        lines.append("## 📋 Summary")
        lines.append(f"- **Type**: {incident_data.get('category', 'UNKNOWN')}")
        lines.append(f"- **Severity**: {incident_data.get('severity', 'UNKNOWN')}")
        lines.append(f"- **Status**: {incident_data.get('status', 'UNKNOWN')}")
        lines.append("")
        
        # Root Cause
        lines.append("## 🔍 Root Cause")
        root_causes = incident_data.get("root_causes", [])
        if root_causes:
            for cause in root_causes[:3]:
                lines.append(f"### {cause.get('type', 'Unknown')}")
                lines.append(f"{cause.get('description', 'No description')}")
                lines.append(f"**Confidence**: {cause.get('confidence', 0.0):.0%}")
                lines.append("")
        else:
            lines.append("_(Analysis pending)_")
            lines.append("")
        
        # Evidence
        if context.include_evidence_details:
            lines.append("## 📊 Evidence")
            evidence_list = incident_data.get("evidence", [])
            if evidence_list:
                for item in evidence_list[:context.max_evidence_items]:
                    lines.append(f"- **{item.get('evidence_type', 'Unknown')}**: {item.get('value', 'N/A')}")
            else:
                lines.append("_(No evidence collected)_")
            lines.append("")
        
        # Impact
        lines.append("## 💥 Impact")
        impact = incident_data.get("impact", {})
        if impact:
            for key, value in impact.items():
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        else:
            lines.append("_(Assessment pending)_")
        lines.append("")
        
        # Recommendations
        if context.include_recommendations_details:
            lines.append("## 🛠️ Recommended Actions")
            recommendations = incident_data.get("recommendations", [])
            if recommendations:
                for idx, rec in enumerate(recommendations[:context.max_recommendations], 1):
                    lines.append(f"{idx}. {rec.get('action', 'Unknown')}")
                    lines.append(f"   - Risk: {rec.get('risk', 'UNKNOWN')}")
                    if rec.get('estimated_impact'):
                        lines.append(f"   - Impact: {rec['estimated_impact']}")
            else:
                lines.append("_(No recommendations)_")
            lines.append("")
        
        # Confidence
        lines.append("## 🧠 Confidence")
        lines.append(ConfidenceFormatter.format_confidence(
            incident_data.get("confidence_score", 0.0)
        ))
        lines.append("")
        
        return "\n".join(lines)
    
    # =====================================================================
    # Slack Rendering
    # =====================================================================
    
    def _render_slack(
        self,
        incident_data: Dict[str, Any],
        context: RenderContext,
    ) -> str:
        """Render for Slack message format."""
        lines = []
        
        severity = incident_data.get("severity", "UNKNOWN")
        severity_emoji = self._get_severity_emoji(severity)
        incident_id = incident_data.get("incident_id", "unknown")
        
        # Header block
        lines.append(f"{severity_emoji} *Incident Alert: {incident_id}*")
        lines.append("")
        
        # Quick summary
        category = incident_data.get("category", "UNKNOWN").replace("_", " ")
        lines.append(f"*Category*: {category}")
        lines.append(f"*Status*: {incident_data.get('status', 'UNKNOWN')}")
        lines.append(f"*Severity*: {severity}")
        lines.append("")
        
        # Root cause
        root_causes = incident_data.get("root_causes", [])
        if root_causes:
            lines.append(f"*Root Cause*: {root_causes[0].get('type', 'Unknown').replace('_', ' ')}")
            lines.append(f"_{root_causes[0].get('description', '')}_")
            lines.append("")
        
        # Impact
        impact = incident_data.get("impact", {})
        if impact:
            lines.append(f"*Impact*: {impact.get('estimated_impact', 'TBD')}")
            lines.append("")
        
        # Actions
        recommendations = incident_data.get("recommendations", [])
        if recommendations:
            lines.append("*Recommended Actions*:")
            for rec in recommendations[:3]:
                action = rec.get("action", "Unknown")
                risk = rec.get("risk", "UNKNOWN")
                lines.append(f"• {action} ({risk})")
        
        # Confidence
        confidence = incident_data.get("confidence_score", 0.0)
        lines.append("")
        lines.append(f"*Confidence*: {confidence:.0%}")
        
        return "\n".join(lines)
    
    # =====================================================================
    # Email Rendering
    # =====================================================================
    
    def _render_email(
        self,
        incident_data: Dict[str, Any],
        context: RenderContext,
    ) -> str:
        """Render for HTML email."""
        incident_id = incident_data.get("incident_id", "unknown")
        severity = incident_data.get("severity", "UNKNOWN")
        confidence = incident_data.get("confidence_score", 0.0)
        
        root_causes = incident_data.get("root_causes", [])
        root_cause_html = ""
        if root_causes:
            cause = root_causes[0]
            root_cause_html = f"""
            <h3>{cause.get('type', 'Unknown').replace('_', ' ')}</h3>
            <p>{cause.get('description', '')}</p>
            """
        
        recommendations = incident_data.get("recommendations", [])
        recommendations_html = ""
        if recommendations:
            items = "".join([
                f"<li>{rec.get('action', 'Unknown')} (Risk: {rec.get('risk', 'UNKNOWN')})</li>"
                for rec in recommendations[:context.max_recommendations]
            ])
            recommendations_html = f"<ol>{items}</ol>"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                h1 {{ color: #d32f2f; }}
                h2 {{ color: #1976d2; border-bottom: 2px solid #1976d2; padding-bottom: 5px; }}
                .alert {{ padding: 15px; background-color: #fff3cd; border-left: 4px solid #ffc107; }}
                .severity {{ padding: 10px; border-radius: 5px; color: white; }}
                .severity.critical {{ background-color: #d32f2f; }}
                .severity.high {{ background-color: #f57c00; }}
                .metadata {{ font-size: 0.9em; color: #666; margin-top: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <h1>🚨 Incident Alert: {incident_id}</h1>
            
            <div class="alert">
                <strong>Immediate Action May Be Required</strong>
            </div>
            
            <h2>Summary</h2>
            <table>
                <tr>
                    <th>Property</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Category</td>
                    <td>{incident_data.get('category', 'UNKNOWN')}</td>
                </tr>
                <tr>
                    <td>Severity</td>
                    <td><span class="severity {severity.lower()}">{severity}</span></td>
                </tr>
                <tr>
                    <td>Status</td>
                    <td>{incident_data.get('status', 'UNKNOWN')}</td>
                </tr>
                <tr>
                    <td>Confidence</td>
                    <td>{confidence:.0%}</td>
                </tr>
            </table>
            
            <h2>Root Cause</h2>
            {root_cause_html if root_cause_html else '<p><em>Analysis pending</em></p>'}
            
            <h2>Recommended Actions</h2>
            {recommendations_html if recommendations_html else '<p><em>No recommendations</em></p>'}
            
            <div class="metadata">
                <p>Generated: {incident_data.get('generated_at', datetime.utcnow().isoformat())}</p>
                <p>Agent: pg-agent</p>
            </div>
        </body>
        </html>
        """
        
        return html.strip()
    
    # =====================================================================
    # Helper Methods
    # =====================================================================
    
    @staticmethod
    def _get_severity_emoji(severity: str) -> str:
        """Get emoji for severity level."""
        if "CRITICAL" in severity:
            return "🔴"
        elif "HIGH" in severity:
            return "🟠"
        elif "MEDIUM" in severity:
            return "🟡"
        elif "LOW" in severity:
            return "🟢"
        return "⚪"
    
    @staticmethod
    def _get_status_emoji(status: str) -> str:
        """Get emoji for status."""
        status_map = {
            "DETECTED": "🔍",
            "INVESTIGATING": "🔎",
            "IDENTIFIED": "✅",
            "MONITORING": "👁️",
            "RESOLVED": "✔️",
            "CLOSED": "📦",
        }
        return status_map.get(status.upper(), "❓")
    
    @staticmethod
    def _get_risk_emoji(risk: str) -> str:
        """Get emoji for risk level."""
        risk_map = {
            "LOW": "🟢",
            "MEDIUM": "🟡",
            "HIGH": "🟠",
            "CRITICAL": "🔴",
        }
        return risk_map.get(risk.upper(), "❓")


# =========================================================================
# Integration Helper Functions
# =========================================================================

def render_incident_to_string(
    incident_data: Dict[str, Any],
    format: RenderFormat = RenderFormat.TERMINAL,
    **kwargs
) -> str:
    """
    Convenience function to render incident to string.
    
    Args:
        incident_data: Incident dictionary
        format: Output format
        **kwargs: Additional RenderContext options
        
    Returns:
        Formatted incident report
    """
    context = RenderContext(format=format, **kwargs)
    renderer = IncidentRenderer()
    return renderer.render(incident_data, context)


def render_incident_to_file(
    incident_data: Dict[str, Any],
    output_path: str,
    format: RenderFormat = RenderFormat.MARKDOWN,
    **kwargs
) -> str:
    """
    Render incident to file.
    
    Args:
        incident_data: Incident dictionary
        output_path: Path to write report
        format: Output format
        **kwargs: Additional RenderContext options
        
    Returns:
        Path to generated file
    """
    content = render_incident_to_string(incident_data, format, **kwargs)
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    
    return str(path.resolve())


# =========================================================================
# Example Usage
# =========================================================================

if __name__ == "__main__":
    # Example incident data
    sample_incident = {
        "incident_id": "INC-20260128-001",
        "category": "QUERY_PERFORMANCE",
        "severity": "P2_HIGH",
        "status": "INVESTIGATING",
        "timestamp": datetime.utcnow().isoformat(),
        "confidence_score": 0.78,
        "root_causes": [
            {
                "type": "MISSING_INDEX",
                "description": "Table users missing index on user_email column used in WHERE clause",
                "confidence": 0.85,
            },
            {
                "type": "STALE_STATISTICS",
                "description": "Statistics on users table haven't been updated in 5 days",
                "confidence": 0.72,
            },
        ],
        "evidence": [
            {
                "evidence_type": "sequential_scan_detected",
                "value": "Sequential scan on users (1.2M rows) taking 8.5s",
                "source": "explain_analyze",
                "confidence": 0.95,
            },
            {
                "evidence_type": "missing_index",
                "value": "user_email column frequently used in filters",
                "source": "query_analysis",
                "confidence": 0.88,
            },
            {
                "evidence_type": "stale_statistics",
                "value": "Last ANALYZE: 5 days ago, rows inserted: 50K since then",
                "source": "pg_stat_user_tables",
                "confidence": 0.92,
            },
        ],
        "impact": {
            "affected_queries": 24,
            "performance_degradation": "5-8 seconds (300% slower than baseline)",
            "estimated_impact": "~15 users affected during peak hours",
            "business_context": "Impacts user authentication flow",
        },
        "recommendations": [
            {
                "action": "CREATE INDEX idx_users_email ON users(user_email)",
                "risk": "LOW",
                "estimated_impact": "Expected 95% improvement",
                "estimated_time": "< 5 minutes",
            },
            {
                "action": "ANALYZE users; VACUUM users;",
                "risk": "LOW",
                "estimated_impact": "Update statistics, enable better plans",
                "estimated_time": "2-3 minutes",
            },
        ],
        "confidence_breakdown": {
            "base_confidence": 0.82,
            "completeness": 0.95,
            "agreement": 1.1,
            "freshness": 0.98,
        },
        "generated_at": datetime.utcnow().isoformat(),
        "agent_version": "1.0.0",
        "caveats": [
            "Analysis based on last 10 minutes of metrics",
            "Recommendations assume production environment",
        ],
    }
    
    # Example: Terminal rendering
    print("\n" + "=" * 80)
    print("TERMINAL OUTPUT")
    print("=" * 80 + "\n")
    terminal_output = render_incident_to_string(
        sample_incident,
        format=RenderFormat.TERMINAL,
    )
    print(terminal_output)
    
    # Example: Markdown rendering
    print("\n" + "=" * 80)
    print("MARKDOWN OUTPUT (snippet)")
    print("=" * 80 + "\n")
    markdown_output = render_incident_to_string(
        sample_incident,
        format=RenderFormat.MARKDOWN,
    )
    print(markdown_output[:500] + "...")
    
    # Example: Slack rendering
    print("\n" + "=" * 80)
    print("SLACK OUTPUT")
    print("=" * 80 + "\n")
    slack_output = render_incident_to_string(
        sample_incident,
        format=RenderFormat.SLACK,
    )
    print(slack_output)
    
    print("\n✅ Incident renderer examples complete")
