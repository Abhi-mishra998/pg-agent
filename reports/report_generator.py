#!/usr/bin/env python3
"""
ReportGenerator - HTML Report Generation

Generates formatted HTML incident reports using Jinja2 templates.
Falls back to a simple renderer if Jinja2 is unavailable.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


class ReportGenerator:
    """
    Generates HTML reports from analysis results.

    Design principles:
    - Pure utility (no env loading, no side effects)
    - Explicit errors
    - Deterministic output
    - Safe fallbacks
    """

    def __init__(
        self,
        template_dir: Optional[str] = None,
        default_template: str = "incident_report.html",
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.template_dir = (
            Path(template_dir)
            if template_dir
            else Path(__file__).parent / "templates"
        )

        self.default_template = default_template
        self.template_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        data: Dict[str, Any],
        output_dir: str = "data/output/",
        template: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate an HTML report.

        Returns:
            Absolute path to generated report
        """
        if not isinstance(data, dict):
            raise TypeError("Report data must be a dictionary")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = filename or self._default_filename()
        template_name = template or self.default_template

        enriched_data = self._enrich_metadata(data)

        html = self._render_template(template_name, enriched_data)

        output_file = output_path / filename
        output_file.write_text(html, encoding="utf-8")

        self.logger.info("Report generated: %s", output_file.resolve())
        return str(output_file.resolve())

    # ------------------------------------------------------------------
    # Template Rendering
    # ------------------------------------------------------------------

    def _render_template(self, template_name: str, data: Dict[str, Any]) -> str:
        try:
            from jinja2 import Environment, FileSystemLoader, select_autoescape

            env = Environment(
                loader=FileSystemLoader(self.template_dir),
                autoescape=select_autoescape(["html", "xml"]),
            )

            if template_name not in env.list_templates():
                raise FileNotFoundError(
                    f"Template '{template_name}' not found in {self.template_dir}"
                )

            template = env.get_template(template_name)
            return template.render(**data)

        except ImportError:
            self.logger.warning("Jinja2 not installed — using simple renderer")
            return self._simple_render(data)

        except Exception as exc:
            self.logger.error("Template rendering failed: %s", exc)
            return self._error_html(str(exc), data)

    # ------------------------------------------------------------------
    # Render Fallbacks
    # ------------------------------------------------------------------

    def _simple_render(self, data: Dict[str, Any]) -> str:
        title = data.get("title", "Report")
        content = data.get("content", "No content provided.")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
    pre {{ background: #f4f4f4; padding: 15px; }}
    .meta {{ color: #666; font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="meta">Generated at: {data.get("generated_at")}</p>
  <pre>{content}</pre>
</body>
</html>
"""

    def _error_html(self, error: str, data: Dict[str, Any]) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Report Error</title>
</head>
<body>
  <h1>Report Generation Error</h1>
  <p><strong>Error:</strong> {error}</p>
  <pre>{data}</pre>
</body>
</html>
"""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_filename(self) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"report_{ts}.html"

    def _enrich_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(data)
        enriched.setdefault("generated_at", datetime.utcnow().isoformat())
        enriched.setdefault("title", "Analysis Report")
        return enriched

    # ------------------------------------------------------------------
    # Template Utilities
    # ------------------------------------------------------------------

    def list_templates(self) -> List[str]:
        """Return template filenames (not paths)."""
        if not self.template_dir.exists():
            return []
        return [p.name for p in self.template_dir.glob("*.html")]

    def get_template_path(self, template_name: str) -> Optional[Path]:
        path = self.template_dir / template_name
        return path if path.exists() else None


# ----------------------------------------------------------------------
# Local test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    generator = ReportGenerator()

    test_data = {
        "title": "Test Report",
        "content": "This is a test report generated by pg-agent.",
        "model": "llama3.1:8b",
    }

    output = generator.generate(test_data)
    print(f"\n✅ Test report generated: {output}")
    print("Available templates:", generator.list_templates())