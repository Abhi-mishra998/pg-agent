#!/usr/bin/env python3
"""
core/pipeline.py

pg-agent Pipeline Orchestration

This module defines the core processing pipeline that orchestrates:
1. Signal Generation (SignalEngine)
2. Evidence Building (EvidenceBuilder)
3. Root Cause Analysis (RootCauseEngine)
4. Recommendation Generation (PgRecommender)
5. Output Rendering

The Pipeline class provides a unified interface for processing incident data
through the entire analysis flow.

Design Principles:
- Type-safe composition with Protocol definitions
- Clear separation of concerns
- Extensible stage implementations
- Built-in timing and metrics
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING

# Import types for Protocol definitions
if TYPE_CHECKING:
    from core.incident import Incident, IncidentMetadata
    from signals.signal_engine import SignalResult, Signal
    from signals.evidence_builder import EvidenceCollection
    from signals.root_cause_engine import RootCauseResult
    from recommendations.pg_recommender import RecommendationReport
    from reports.report_generator import ReportGenerator


# =====================================================================
# PROTOCOLS (Stage Contracts)
# =====================================================================

class SignalAnalyzer(Protocol):
    """Protocol for signal analysis engine."""
    
    def process(self, data: Any) -> "SignalResult":
        """
        Process raw data and produce signals.
        
        Args:
            data: Raw incident data (dict, string, JSON, etc.)
            
        Returns:
            SignalResult with detected signals and analysis
        """
        ...


class EvidenceCollector(Protocol):
    """Protocol for evidence collection engine."""
    
    def build(
        self,
        signal_result: "SignalResult",
        context: Optional[Dict[str, Any]] = None,
    ) -> "EvidenceCollection":
        """
        Build evidence collection from signals.
        
        Args:
            signal_result: Output from SignalAnalyzer
            context: Optional contextual information
            
        Returns:
            EvidenceCollection with structured evidence
        """
        ...


class RootCauseAnalyzer(Protocol):
    """Protocol for root cause analysis engine."""
    
    def analyze(
        self,
        signal_result: "SignalResult",
        evidence_collection: "EvidenceCollection",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, "RootCauseResult"]:
        """
        Analyze signals and evidence to identify root causes.
        
        Args:
            signal_result: Output from SignalAnalyzer
            evidence_collection: Output from EvidenceCollector
            context: Optional contextual information
            
        Returns:
            Dictionary mapping RootCauseCategory to RootCauseResult
        """
        ...


class RecommenderEngine(Protocol):
    """Protocol for recommendation engine."""
    
    def recommend(
        self,
        signal_result: "SignalResult",
        evidence: "EvidenceCollection",
    ) -> "RecommendationReport":
        """
        Generate recommendations based on signals and evidence.
        
        Args:
            signal_result: Output from SignalAnalyzer
            evidence: Output from EvidenceCollector
            
        Returns:
            RecommendationReport with actionable recommendations
        """
        ...


class OutputFormatter(Protocol):
    """Protocol for output formatting."""
    
    def format(
        self,
        report: "RecommendationReport",
        format_type: str = "terminal",
    ) -> str:
        """
        Format recommendation report for output.
        
        Args:
            report: RecommendationReport to format
            format_type: Output format (terminal, markdown, json, html)
            
        Returns:
            Formatted report as string
        """
        ...


# =====================================================================
# DATA MODELS
# =====================================================================

@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline execution."""
    total_time_ms: float = 0.0
    signal_time_ms: float = 0.0
    evidence_time_ms: float = 0.0
    root_cause_time_ms: float = 0.0
    recommendation_time_ms: float = 0.0
    formatting_time_ms: float = 0.0
    
    signal_count: int = 0
    evidence_count: int = 0
    root_cause_count: int = 0
    recommendation_count: int = 0
    
    confidence: float = 0.0
    risk_level: str = "unknown"
    
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_time_ms": self.total_time_ms,
            "signal_time_ms": self.signal_time_ms,
            "evidence_time_ms": self.evidence_time_ms,
            "root_cause_time_ms": self.root_cause_time_ms,
            "recommendation_time_ms": self.recommendation_time_ms,
            "formatting_time_ms": self.formatting_time_ms,
            "signal_count": self.signal_count,
            "evidence_count": self.evidence_count,
            "root_cause_count": self.root_cause_count,
            "recommendation_count": self.recommendation_count,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "timestamp": self.timestamp,
        }


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    success: bool
    metrics: PipelineMetrics
    signal_result: Optional["SignalResult"] = None
    evidence_collection: Optional["EvidenceCollection"] = None
    root_cause_results: Optional[Dict[str, "RootCauseResult"]] = None
    recommendation_report: Optional["RecommendationReport"] = None
    formatted_output: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "metrics": self.metrics.to_dict(),
            "signal_count": self.metrics.signal_count,
            "evidence_count": self.metrics.evidence_count,
            "root_cause_count": self.metrics.root_cause_count,
            "recommendation_count": self.metrics.recommendation_count,
            "confidence": self.metrics.confidence,
            "risk_level": self.metrics.risk_level,
            "error": self.error,
            "output": self.formatted_output,
        }


# =====================================================================
# PIPELINE CLASS
# =====================================================================

class Pipeline:
    """
    pg-agent Pipeline Orchestrator.
    
    Coordinates the complete analysis flow:
    Signal → Evidence → Root Cause → Recommendation → Output
    
    Usage:
        pipeline = Pipeline()
        result = pipeline.run(data)
    """
    
    def __init__(
        self,
        signal_analyzer: Optional[SignalAnalyzer] = None,
        evidence_collector: Optional[EvidenceCollector] = None,
        root_cause_analyzer: Optional[RootCauseAnalyzer] = None,
        recommender: Optional[RecommenderEngine] = None,
        output_formatter: Optional[OutputFormatter] = None,
        log_level: str = "INFO",
    ):
        """
        Initialize the pipeline with stage implementations.
        
        Args:
            signal_analyzer: SignalAnalyzer implementation (defaults to SignalEngine)
            evidence_collector: EvidenceCollector implementation (defaults to EvidenceBuilder)
            root_cause_analyzer: RootCauseAnalyzer implementation (defaults to RootCauseEngine)
            recommender: RecommenderEngine implementation (defaults to PgRecommender)
            output_formatter: OutputFormatter implementation
            log_level: Logging level
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # Initialize or use provided stages
        self._init_stages(
            signal_analyzer=signal_analyzer,
            evidence_collector=evidence_collector,
            root_cause_analyzer=root_cause_analyzer,
            recommender=recommender,
            output_formatter=output_formatter,
        )
        
        self.logger.info("Pipeline initialized successfully")
    
    def _init_stages(
        self,
        signal_analyzer: Optional[SignalAnalyzer],
        evidence_collector: Optional[EvidenceCollector],
        root_cause_analyzer: Optional[RootCauseAnalyzer],
        recommender: Optional[RecommenderEngine],
        output_formatter: Optional[OutputFormatter],
    ) -> None:
        """Initialize all pipeline stages."""
        
        # Signal Analyzer
        if signal_analyzer is not None:
            self.signal_analyzer = signal_analyzer
        else:
            from signals.signal_engine import SignalEngine
            self.signal_analyzer = SignalEngine()
            self.logger.info("Using default SignalEngine")
        
        # Evidence Collector
        if evidence_collector is not None:
            self.evidence_collector = evidence_collector
        else:
            from signals.evidence_builder import EvidenceBuilder
            self.evidence_collector = EvidenceBuilder()
            self.logger.info("Using default EvidenceBuilder")
        
        # Root Cause Analyzer
        if root_cause_analyzer is not None:
            self.root_cause_analyzer = root_cause_analyzer
        else:
            from signals.root_cause_engine import RootCauseEngine
            self.root_cause_analyzer = RootCauseEngine()
            self.logger.info("Using default RootCauseEngine")
        
        # Recommender
        if recommender is not None:
            self.recommender = recommender
        else:
            from recommendations.pg_recommender import PgRecommender
            self.recommender = PgRecommender()
            self.logger.info("Using default PgRecommender")
        
        # Output Formatter
        if output_formatter is not None:
            self.output_formatter = output_formatter
        else:
            from core.output_formatter import OutputFormatter
            self.output_formatter = OutputFormatter()
            self.logger.info("Using default OutputFormatter")
    
    def run(
        self,
        data: Any,
        context: Optional[Dict[str, Any]] = None,
        output_format: str = "terminal",
    ) -> PipelineResult:
        """
        Execute the complete pipeline.
        
        Args:
            data: Raw input data (text, JSON, file path, etc.)
            context: Optional context dictionary
            output_format: Desired output format (terminal, markdown, json, html)
            
        Returns:
            PipelineResult with all outputs and metrics
        """
        start_time = time.time()
        metrics = PipelineMetrics()
        context = context or {}
        
        self.logger.info("Starting pipeline execution")
        
        try:
            # Stage 1: Signal Analysis
            signal_start = time.time()
            self.logger.info("Stage 1: Signal Analysis")
            
            signal_result = self.signal_analyzer.process(data)
            metrics.signal_count = len(signal_result.signals)
            metrics.signal_time_ms = (time.time() - signal_start) * 1000
            
            self.logger.info(
                f"Signal analysis complete: {metrics.signal_count} signals in "
                f"{metrics.signal_time_ms:.2f}ms"
            )
            
            # Stage 2: Evidence Building
            evidence_start = time.time()
            self.logger.info("Stage 2: Evidence Building")
            
            evidence_collection = self.evidence_collector.build(
                signal_result, context
            )
            metrics.evidence_count = len(evidence_collection.evidence)
            metrics.confidence = evidence_collection.overall_confidence
            metrics.evidence_time_ms = (time.time() - evidence_start) * 1000
            
            self.logger.info(
                f"Evidence building complete: {metrics.evidence_count} evidence items "
                f"in {metrics.evidence_time_ms:.2f}ms, confidence: {metrics.confidence}"
            )
            
            # Stage 3: Root Cause Analysis
            root_cause_start = time.time()
            self.logger.info("Stage 3: Root Cause Analysis")
            
            root_cause_results = self.root_cause_analyzer.analyze(
                signal_result=signal_result,
                evidence_collection=evidence_collection,
                context=context,
            )
            
            # Count likely root causes
            likely_causes = [
                rc for rc in root_cause_results.values()
                if rc.is_likely_cause
            ]
            metrics.root_cause_count = len(likely_causes)
            metrics.root_cause_time_ms = (time.time() - root_cause_start) * 1000
            
            self.logger.info(
                f"Root cause analysis complete: {metrics.root_cause_count} likely causes "
                f"in {metrics.root_cause_time_ms:.2f}ms"
            )
            
            # Stage 4: Recommendation Generation
            rec_start = time.time()
            self.logger.info("Stage 4: Recommendation Generation")
            
            recommendation_report = self.recommender.recommend(
                signal_result=signal_result,
                evidence=evidence_collection,
            )
            metrics.recommendation_count = len(recommendation_report.recommendations)
            metrics.risk_level = recommendation_report.risk_level
            metrics.recommendation_time_ms = (time.time() - rec_start) * 1000
            
            self.logger.info(
                f"Recommendation generation complete: {metrics.recommendation_count} "
                f"recommendations in {metrics.recommendation_time_ms:.2f}ms, "
                f"risk: {metrics.risk_level}"
            )
            
            # Stage 5: Output Formatting
            format_start = time.time()
            self.logger.info("Stage 5: Output Formatting")

            # Extract root causes from results
            root_causes = []
            for category, rc_result in root_cause_results.items():
                if rc_result.is_likely_cause:
                    factors = ", ".join(rc_result.contributing_factors[:2]) if rc_result.contributing_factors else "Analysis in progress"
                    root_causes.append(f"{category.value}: {factors}")

            # Format output based on requested format type
            if output_format == "json":
                formatted_output = self.output_formatter.format_json(
                    signals=signal_result.signals,
                    evidence=evidence_collection,
                    root_causes=root_causes,
                    recommendations=recommendation_report.recommendations,
                    risk_level=recommendation_report.risk_level,
                    processing_time=metrics.total_time_ms,
                )
            elif output_format == "markdown":
                formatted_output = self.output_formatter.format_markdown(
                    signals=signal_result.signals,
                    evidence=evidence_collection,
                    root_causes=root_causes,
                    recommendations=recommendation_report.recommendations,
                    risk_level=recommendation_report.risk_level,
                    processing_time=metrics.total_time_ms,
                )
            else:
                # Default to terminal format
                formatted_output = self.output_formatter.format(
                    signals=signal_result.signals,
                    evidence=evidence_collection,
                    root_causes=root_causes,
                    recommendations=recommendation_report.recommendations,
                    risk_level=recommendation_report.risk_level,
                    processing_time=metrics.total_time_ms,
                )
            metrics.formatting_time_ms = (time.time() - format_start) * 1000
            
            self.logger.info(
                f"Output formatting complete in {metrics.formatting_time_ms:.2f}ms"
            )
            
            # Calculate total time
            metrics.total_time_ms = (time.time() - start_time) * 1000
            
            self.logger.info(
                f"Pipeline completed in {metrics.total_time_ms:.2f}ms total"
            )
            
            return PipelineResult(
                success=True,
                metrics=metrics,
                signal_result=signal_result,
                evidence_collection=evidence_collection,
                root_cause_results=root_cause_results,
                recommendation_report=recommendation_report,
                formatted_output=formatted_output,
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            metrics.total_time_ms = (time.time() - start_time) * 1000
            
            return PipelineResult(
                success=False,
                metrics=metrics,
                error=str(e),
            )
    
    def run_simple(
        self,
        data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run pipeline and return simplified result dictionary.
        
        This is a convenience method for quick access to results.
        
        Args:
            data: Raw input data
            context: Optional context
            
        Returns:
            Dictionary with keys: success, signals, evidence, recommendations, metrics
        """
        result = self.run(data, context)
        
        # Extract root causes
        root_causes = []
        if result.root_cause_results:
            for category, rc_result in result.root_cause_results.items():
                if rc_result.is_likely_cause:
                    root_causes.append({
                        "category": category.value if hasattr(category, 'value') else str(category),
                        "confidence": rc_result.confidence,
                        "recommendations": rc_result.recommendations[:2],
                    })
        
        return {
            "success": result.success,
            "signals": [
                {"name": s.name, "severity": s.severity, "data": s.data}
                for s in (result.signal_result.signals if result.signal_result else [])
            ],
            "evidence_count": result.metrics.evidence_count,
            "confidence": result.metrics.confidence,
            "root_causes": root_causes,
            "recommendations": [
                {"title": r.title, "severity": r.severity, "description": r.description}
                for r in (result.recommendation_report.recommendations if result.recommendation_report else [])
            ],
            "risk_level": result.metrics.risk_level,
            "processing_time_ms": result.metrics.total_time_ms,
            "error": result.error,
        }
    
    def analyze_incident(
        self,
        data: Any,
    ) -> Dict[str, Any]:
        """
        Run full incident analysis pipeline.
        
        Args:
            data: Incident data (JSON dict, text, etc.)
            
        Returns:
            Complete analysis result with all details
        """
        result = self.run(data, output_format="dict")
        
        # Format root causes for output
        root_causes_output = {}
        if result.root_cause_results:
            for category, rc_result in result.root_cause_results.items():
                cat_key = category.value if hasattr(category, 'value') else str(category)
                root_causes_output[cat_key] = rc_result.to_dict()
        
        return {
            "success": result.success,
            "incident_id": "",
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": {
                "signals": result.signal_result.to_dict() if result.signal_result else None,
                "evidence": {
                    "count": result.metrics.evidence_count,
                    "confidence": result.metrics.confidence,
                    "types": result.evidence_collection.evidence_types if result.evidence_collection else {},
                } if result.evidence_collection else None,
                "root_causes": root_causes_output,
                "recommendations": result.recommendation_report.to_dict() if result.recommendation_report else None,
            },
            "metrics": result.metrics.to_dict(),
            "error": result.error,
        }


# =====================================================================
# FACTORY FUNCTIONS
# =====================================================================

def create_default_pipeline(log_level: str = "INFO") -> Pipeline:
    """
    Create pipeline with all default stage implementations.
    
    Args:
        log_level: Logging level
        
    Returns:
        Configured Pipeline instance
    """
    return Pipeline(log_level=log_level)


def create_lightweight_pipeline(log_level: str = "INFO") -> Pipeline:
    """
    Create pipeline optimized for speed (minimal processing).
    
    Args:
        log_level: Logging level
        
    Returns:
        Lightweight Pipeline instance
    """
    from signals.signal_engine import SignalEngine
    from signals.evidence_builder import EvidenceBuilder
    from recommendations.pg_recommender import PgRecommender
    
    signal_analyzer = SignalEngine(log_level=log_level)
    evidence_collector = EvidenceBuilder(
        min_confidence=0.7,  # Higher threshold = faster
        correlation_enabled=False,  # Skip correlation for speed
    )
    recommender = PgRecommender()
    
    return Pipeline(
        signal_analyzer=signal_analyzer,
        evidence_collector=evidence_collector,
        recommender=recommender,
        log_level=log_level,
    )


# =====================================================================
# EXAMPLE USAGE
# =====================================================================

if __name__ == "__main__":
    # Example: Running the pipeline
    pipeline = create_default_pipeline()
    
    # Test with sample data
    sample_data = """
    ERROR: deadlock detected
    DETAIL: Process 12345 waits for ShareLock on transaction 67890
    HINT: Review application code for proper transaction handling
    """
    
    print("Running pipeline with sample data...")
    result = pipeline.run(sample_data)
    
    print(f"\nSuccess: {result.success}")
    print(f"Signals: {result.metrics.signal_count}")
    print(f"Evidence: {result.metrics.evidence_count}")
    print(f"Recommendations: {result.metrics.recommendation_count}")
    print(f"Confidence: {result.metrics.confidence}")
    print(f"Risk Level: {result.metrics.risk_level}")
    print(f"Time: {result.metrics.total_time_ms:.2f}ms")
    
    if result.error:
        print(f"Error: {result.error}")

