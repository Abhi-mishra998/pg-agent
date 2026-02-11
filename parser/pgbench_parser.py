#!/usr/bin/env python3
"""
pgbench_parser.py

Purpose:
- Parse pgbench reports (text / JSON-like)
- Extract TPS, latency, failures
- Generate structured performance insights
- Feed SignalEngine with meaningful data

Designed for:
- Incident analysis
- Performance regression detection
- Capacity planning signals
"""

import re
import statistics
from typing import Dict, Any, List, Optional


class PgbenchParser:
    """
    Parses pgbench output and produces normalized performance metrics.
    """

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def parse(self, content: str) -> Dict[str, Any]:
        """
        Entry point.

        Args:
            content: Raw pgbench output (file content)

        Returns:
            Normalized pgbench metrics
        """
        metrics = {
            "tps": self._extract_tps(content),
            "latency_ms": self._extract_latency(content),
            "errors": self._extract_errors(content),
            "scale_factor": self._extract_scale(content),
            "clients": self._extract_clients(content),
        }

        metrics["analysis"] = self._analyze(metrics)
        metrics["recommendations"] = self._recommend(metrics)

        return metrics

    # ---------------------------------------------------------
    # Extraction helpers
    # ---------------------------------------------------------

    def _extract_tps(self, text: str) -> Optional[float]:
        """
        Extract TPS (transactions per second).
        """
        match = re.search(r"tps\s*=\s*([\d\.]+)", text, re.IGNORECASE)
        return float(match.group(1)) if match else None

    def _extract_latency(self, text: str) -> Dict[str, Optional[float]]:
        """
        Extract latency stats.
        """
        avg = self._find_float(r"latency average\s*=\s*([\d\.]+)\s*ms", text)
        stddev = self._find_float(r"latency stddev\s*=\s*([\d\.]+)\s*ms", text)

        return {
            "avg_ms": avg,
            "stddev_ms": stddev,
        }

    def _extract_errors(self, text: str) -> int:
        """
        Count errors or failed transactions.
        """
        return len(re.findall(r"ERROR|FATAL|timeout", text, re.IGNORECASE))

    def _extract_scale(self, text: str) -> Optional[int]:
        match = re.search(r"scaling factor:\s*(\d+)", text, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _extract_clients(self, text: str) -> Optional[int]:
        match = re.search(r"number of clients:\s*(\d+)", text, re.IGNORECASE)
        return int(match.group(1)) if match else None

    # ---------------------------------------------------------
    # Analysis Logic
    # ---------------------------------------------------------

    def _analyze(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpret pgbench metrics.
        """
        analysis = {
            "status": "healthy",
            "issues": [],
        }

        tps = metrics["tps"]
        latency = metrics["latency_ms"]["avg_ms"]
        errors = metrics["errors"]

        if tps is not None and tps < 100:
            analysis["status"] = "degraded"
            analysis["issues"].append("Low TPS detected")

        if latency is not None and latency > 200:
            analysis["status"] = "degraded"
            analysis["issues"].append("High latency")

        if errors > 0:
            analysis["status"] = "critical"
            analysis["issues"].append("Errors in workload")

        return analysis

    # ---------------------------------------------------------
    # Recommendation Engine
    # ---------------------------------------------------------

    def _recommend(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations.
        """
        recommendations = []

        tps = metrics["tps"]
        latency = metrics["latency_ms"]["avg_ms"]
        clients = metrics["clients"]
        scale = metrics["scale_factor"]

        if latency and latency > 200:
            recommendations.append(
                "Investigate slow queries using pg_stat_statements"
            )
            recommendations.append(
                "Check missing indexes on frequently updated tables"
            )

        if tps and tps < 100:
            recommendations.append(
                "Increase shared_buffers and work_mem"
            )
            recommendations.append(
                "Review connection pooling (PgBouncer recommended)"
            )

        if clients and clients > 50:
            recommendations.append(
                "High client count detected; connection pooling advised"
            )

        if scale and scale > 100:
            recommendations.append(
                "Large scale factor — ensure autovacuum is tuned"
            )

        if not recommendations:
            recommendations.append(
                "No immediate performance issues detected"
            )

        return recommendations

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    def _find_float(self, pattern: str, text: str) -> Optional[float]:
        match = re.search(pattern, text, re.IGNORECASE)
        return float(match.group(1)) if match else None