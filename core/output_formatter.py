#!/usr/bin/env python3
"""
OutputFormatter - Terminal Output Formatting

Formats incident analysis results into readable terminal output.
Supports multiple output formats: terminal, markdown, json.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional


class OutputFormatter:
    """
    Formats incident analysis results for terminal output.
    
    Design principles:
    - Pure utility (no env loading, no side effects)
    - Clear, readable output
    - Support for multiple formats
    """

    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    def _get_evidence_count(self, evidence: Any) -> int:
        """Safely get evidence count, handling different evidence types."""
        if evidence is None:
            return 0
        if hasattr(evidence, '__len__'):
            return len(evidence)
        elif hasattr(evidence, 'evidence'):
            return len(evidence.evidence)
        elif hasattr(evidence, 'evidence_list'):
            return len(evidence.evidence_list)
        elif hasattr(evidence, 'items'):
            return len(evidence.items)
        return 0

    def _get_evidence_confidence(self, evidence: Any) -> float:
        """Safely get evidence confidence."""
        if evidence is None:
            return 0.0
        if hasattr(evidence, 'overall_confidence'):
            return evidence.overall_confidence
        elif hasattr(evidence, 'confidence'):
            return getattr(evidence, 'confidence', 0.0)
        return 0.0

    def _signal_to_str(self, signal: Any) -> str:
        """Convert signal to string representation."""
        if hasattr(signal, 'name'):
            return signal.name
        elif hasattr(signal, 'title'):
            return signal.title
        elif hasattr(signal, '__str__'):
            return str(signal)
        return repr(signal)

    def _recommendation_to_str(self, rec: Any) -> str:
        """Convert recommendation to string representation."""
        if hasattr(rec, 'title'):
            return rec.title
        elif hasattr(rec, 'description'):
            return rec.description
        elif hasattr(rec, '__str__'):
            return str(rec)
        return repr(rec)

    def format(
        self,
        signals: List[Any],
        evidence: Any,
        root_causes: List[Any],
        recommendations: List[Any],
        risk_level: str,
        processing_time: float,
    ) -> str:
        """
        Format analysis results into a readable string.
        
        Args:
            signals: List of detected signals
            evidence: Evidence collection object
            root_causes: List of identified root causes
            recommendations: List of recommendations
            risk_level: Risk level string
            processing_time: Processing time in milliseconds
            
        Returns:
            Formatted output string
        """
        lines = []

        lines.append("🧠 INCIDENT ANALYSIS")
        lines.append("-" * 50)

        # Signals section
        if not signals:
            lines.append("✅ No critical signals detected")
        else:
            for s in signals:
                lines.append(f"• {self._signal_to_str(s)}")

        # Evidence section
        evidence_count = self._get_evidence_count(evidence)
        evidence_confidence = self._get_evidence_confidence(evidence)
        lines.append("\n📊 EVIDENCE SUMMARY")
        lines.append(f"• Evidence count : {evidence_count}")
        lines.append(f"• Confidence     : {evidence_confidence}")

        # Root causes section
        lines.append("\n🔍 ROOT CAUSE ANALYSIS")
        if not root_causes:
            lines.append("✅ No root causes identified")
        else:
            for rc in root_causes:
                lines.append(f"• {rc}")

        # Recommendations section
        lines.append("\n🛠️ EXPERT RECOMMENDATIONS")
        if not recommendations:
            lines.append("✅ No action required")
        else:
            for r in recommendations:
                lines.append(f"• {self._recommendation_to_str(r)}")

        # Risk and timing
        lines.append(f"\n📈 Risk Level: {risk_level}")
        lines.append(f"⏱️  Processing Time: {processing_time:.2f}ms")

        return "\n".join(lines)

    def format_terminal(
        self,
        signals: List[Any],
        evidence: Any,
        root_causes: List[Any],
        recommendations: List[Any],
        risk_level: str,
        processing_time: float,
    ) -> str:
        """Format for terminal output (same as default format)."""
        return self.format(
            signals, evidence, root_causes, recommendations, risk_level, processing_time
        )

    def format_markdown(
        self,
        signals: List[Any],
        evidence: Any,
        root_causes: List[Any],
        recommendations: List[Any],
        risk_level: str,
        processing_time: float,
    ) -> str:
        """Format for Markdown output."""
        lines = []
        
        evidence_count = self._get_evidence_count(evidence)
        evidence_confidence = self._get_evidence_confidence(evidence)

        lines.append("## 🧠 INCIDENT ANALYSIS")
        lines.append("")

        # Signals section
        lines.append("### Signals")
        if not signals:
            lines.append("✅ No critical signals detected")
        else:
            for s in signals:
                lines.append(f"- {self._signal_to_str(s)}")
        lines.append("")

        # Evidence section
        lines.append("### 📊 Evidence Summary")
        lines.append(f"- **Evidence count**: {evidence_count}")
        lines.append(f"- **Confidence**: {evidence_confidence}")
        lines.append("")

        # Root causes section
        lines.append("### 🔍 Root Cause Analysis")
        if not root_causes:
            lines.append("✅ No root causes identified")
        else:
            for rc in root_causes:
                lines.append(f"- {rc}")
        lines.append("")

        # Recommendations section
        lines.append("### 🛠️ Expert Recommendations")
        if not recommendations:
            lines.append("✅ No action required")
        else:
            for r in recommendations:
                lines.append(f"- {self._recommendation_to_str(r)}")
        lines.append("")

        # Risk and timing
        lines.append(f"**Risk Level**: {risk_level}")
        lines.append(f"**Processing Time**: {processing_time:.2f}ms")
        lines.append(f"\n*Generated at: {datetime.utcnow().isoformat()}*")

        return "\n".join(lines)

    def format_json(
        self,
        signals: List[Any],
        evidence: Any,
        root_causes: List[Any],
        recommendations: List[Any],
        risk_level: str,
        processing_time: float,
    ) -> str:
        """Format for JSON output."""
        evidence_count = self._get_evidence_count(evidence)
        evidence_confidence = self._get_evidence_confidence(evidence)
        
        output = {
            "incident_analysis": {
                "signals": [self._signal_to_str(s) for s in signals],
                "evidence_summary": {
                    "count": evidence_count,
                    "confidence": evidence_confidence,
                },
                "root_causes": [str(rc) for rc in root_causes],
                "recommendations": [self._recommendation_to_str(r) for r in recommendations],
                "risk_level": risk_level,
                "processing_time_ms": processing_time,
                "timestamp": datetime.utcnow().isoformat(),
            }
        }
        return json.dumps(output, indent=2)

    def format_html(
        self,
        signals: List[Any],
        evidence: Any,
        root_causes: List[Any],
        recommendations: List[Any],
        risk_level: str,
        processing_time: float,
    ) -> str:
        """Format for HTML output."""
        evidence_count = self._get_evidence_count(evidence)
        evidence_confidence = self._get_evidence_confidence(evidence)
        
        # Convert signals to HTML
        signals_html = ""
        if not signals:
            signals_html = "<li class='success'>✅ No critical signals detected</li>"
        else:
            for s in signals:
                signals_html += f"<li>• {self._signal_to_str(s)}</li>"

        # Convert recommendations to HTML
        recs_html = ""
        if not recommendations:
            recs_html = "<li class='success'>✅ No action required</li>"
        else:
            for r in recommendations:
                recs_html += f"<li>• {self._recommendation_to_str(r)}</li>"

        # Risk level color
        risk_color_map = {
            "LOW": "#28a745",
            "MEDIUM": "#ffc107",
            "HIGH": "#fd7e14",
            "CRITICAL": "#dc3545",
        }
        risk_color = risk_color_map.get(risk_level.upper(), "#6c757d")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Incident Analysis Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
    .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
    h2 {{ color: #555; margin-top: 25px; }}
    ul {{ padding-left: 20px; }}
    li {{ margin: 8px 0; }}
    .metric {{ display: inline-block; background: #e9ecef; padding: 8px 15px; border-radius: 4px; margin: 5px; }}
    .risk {{ color: white; padding: 10px 20px; border-radius: 4px; display: inline-block; }}
    .timestamp {{ color: #666; font-size: 0.9em; }}
    .success {{ color: #28a745; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>🧠 INCIDENT ANALYSIS</h1>
    <p class="timestamp">Generated at: {datetime.utcnow().isoformat()}</p>
    
    <h2>Signals</h2>
    <ul>
{signals_html}
    </ul>
    
    <h2>📊 Evidence Summary</h2>
    <div class="metric">Evidence count: {evidence_count}</div>
    <div class="metric">Confidence: {evidence_confidence}</div>
    
    <h2>🔍 Root Causes</h2>
    <ul>
      <li>{len(root_causes)} root causes identified</li>
    </ul>
    
    <h2>🛠️ Expert Recommendations</h2>
    <ul>
{recs_html}
    </ul>
    
    <h2>📈 Risk & Timing</h2>
    <div class="risk" style="background: {risk_color};">Risk Level: {risk_level}</div>
    <div class="metric">Processing Time: {processing_time:.2f}ms</div>
  </div>
</body>
</html>
"""
        return html


# ----------------------------------------------------------------------
# Local test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    formatter = OutputFormatter()

    # Mock data
    signals = ["High CPU usage detected", "Memory pressure warning"]
    evidence_count = 5
    evidence_confidence = 0.85
    root_causes = ["Index fragmentation", "Inefficient query"]
    recommendations = ["Run VACUUM FULL", "Add index on column X"]

    class MockEvidence:
        def __init__(self):
            self.confidence = evidence_confidence
        def __len__(self):
            return evidence_count

    evidence = MockEvidence()

    print("=== Terminal Output ===")
    print(formatter.format_terminal(
        signals, evidence, root_causes, recommendations, "HIGH", 123.45
    ))

    print("\n=== Markdown Output ===")
    print(formatter.format_markdown(
        signals, evidence, root_causes, recommendations, "HIGH", 123.45
    ))

    print("\n=== JSON Output ===")
    print(formatter.format_json(
        signals, evidence, root_causes, recommendations, "HIGH", 123.45
    ))

