#!/usr/bin/env python3
"""
Clarification-Question Framework for PostgreSQL Agent

Design Principles:
1. Precise and minimal questions - only ask when confidence < threshold
2. Questions map to missing evidence - no hallucinations
3. Deterministic - same state produces same questions
4. Evidence-driven - questions are derived from what's missing

Decision Logic:
- Trigger when confidence < 0.75 (Moderate threshold)
- Identify missing required evidence types
- Map evidence types to specific questions
- Prioritize by impact on confidence
- Ask one question at a time

Question Categories:
1. Deployment Changes
2. Performance Baseline
3. Data Operations
4. Maintenance Status
5. Configuration Changes

No LLM hallucinations - all questions are structured templates.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------

class EvidenceType(Enum):
    """Types of evidence that can be missing."""
    DEPLOYMENT_SCHEMA = "deployment_schema"
    PERFORMANCE_BASELINE = "performance_baseline"
    BACKGROUND_JOBS = "background_jobs"
    STATISTICS_MAINTENANCE = "statistics_maintenance"
    CONFIGURATION = "configuration"
    QUERY_METRICS = "query_metrics"
    TABLE_STATS = "table_stats"
    INDEX_HEALTH = "index_health"
    LOCKING = "locking"
    HARDWARE_CAPACITY = "hardware_capacity"


class QuestionPriority(Enum):
    """Priority levels for questions."""
    CRITICAL = 1  # High impact on confidence
    HIGH = 2      # Medium-high impact
    MEDIUM = 3    # Medium impact
    LOW = 4       # Low impact


class QuestionCategory(Enum):
    """Categories of clarification questions."""
    DEPLOYMENT = "DEPLOYMENT"
    PERFORMANCE = "PERFORMANCE"
    DATA_OPERATIONS = "DATA_OPERATIONS"
    MAINTENANCE = "MAINTENANCE"
    CONFIGURATION = "CONFIGURATION"


# ---------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------

@dataclass
class ClarificationAnswer:
    """
    Structured answer to a clarification question.
    
    All answers are structured to avoid free-text ambiguity.
    """
    question_id: str
    category: QuestionCategory
    answer_value: Any  # Structured answer
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    confidence_impact: float = 0.0  # How much this improves confidence
    evidence_generated: List[str] = field(default_factory=list)  # Evidence IDs this creates
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "category": self.category.value,
            "answer_value": self.answer_value,
            "timestamp": self.timestamp,
            "confidence_impact": self.confidence_impact,
            "evidence_generated": self.evidence_generated,
        }


@dataclass
class ClarificationQuestion:
    """
    A structured clarification question.
    
    Questions are precise, minimal, and map to missing evidence.
    """
    question_id: str
    question_text: str
    category: QuestionCategory
    evidence_type: EvidenceType
    priority: QuestionPriority
    missing_evidence_count: int  # How many evidence items of this type are missing
    confidence_threshold: float  # At what confidence level this question triggers
    template_id: str  # Which template generated this
    expected_answer_type: str  # Type of answer expected (yes/no, timestamp, etc.)
    answer_options: List[str] = field(default_factory=list)  # Structured options if applicable
    
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "category": self.category.value,
            "evidence_type": self.evidence_type.value,
            "priority": self.priority.value,
            "missing_evidence_count": self.missing_evidence_count,
            "confidence_threshold": self.confidence_threshold,
            "template_id": self.template_id,
            "expected_answer_type": self.expected_answer_type,
            "answer_options": self.answer_options,
        }
    
    def to_human_readable(self) -> str:
        return f"[{self.category.value}] {self.question_text}"


@dataclass
class ClarificationState:
    """
    Tracks the state of clarification for an incident.
    
    Manages which questions have been asked, answered, and what's remaining.
    """
    questions_asked: List[ClarificationQuestion] = field(default_factory=list)
    answers_received: List[ClarificationAnswer] = field(default_factory=list)
    pending_questions: List[ClarificationQuestion] = field(default_factory=list)
    original_confidence: float = 0.0
    current_confidence: float = 0.0
    target_confidence: float = 0.75  # Target confidence to stop asking
    max_questions: int = 5  # Maximum questions to ask
    questions_asked_count: int = 0
    session_start: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "questions_asked": [q.to_dict() for q in self.questions_asked],
            "answers_received": [a.to_dict() for a in self.answers_received],
            "pending_questions": [q.to_dict() for q in self.pending_questions],
            "original_confidence": self.original_confidence,
            "current_confidence": self.current_confidence,
            "target_confidence": self.target_confidence,
            "max_questions": self.max_questions,
            "questions_asked_count": self.questions_asked_count,
            "session_start": self.session_start,
        }


# ---------------------------------------------------------------------
# Question Templates Registry
# ---------------------------------------------------------------------

class QuestionTemplate:
    """
    A structured question template.
    
    Templates are fixed - no LLM generation. Each template:
    - Has a specific question text
    - Maps to specific evidence type
    - Has structured answer options
    - Has defined confidence impact
    """
    
    def __init__(
        self,
        template_id: str,
        question_text: str,
        evidence_type: EvidenceType,
        category: QuestionCategory,
        priority: QuestionPriority,
        expected_answer_type: str,
        answer_options: List[str],
        confidence_impact: float,
    ):
        self.template_id = template_id
        self.question_text = question_text
        self.evidence_type = evidence_type
        self.category = category
        self.priority = priority
        self.expected_answer_type = expected_answer_type
        self.answer_options = answer_options
        self.confidence_impact = confidence_impact
    
    def generate_question(
        self,
        missing_evidence_count: int = 1,
        confidence_threshold: float = 0.75,
    ) -> ClarificationQuestion:
        """Generate a question from this template."""
        return ClarificationQuestion(
            question_id=f"q_{self.template_id}_{datetime.utcnow().strftime('%H%M%S')}",
            question_text=self.question_text,
            category=self.category,
            evidence_type=self.evidence_type,
            priority=self.priority,
            missing_evidence_count=missing_evidence_count,
            confidence_threshold=confidence_threshold,
            template_id=self.template_id,
            expected_answer_type=self.expected_answer_type,
            answer_options=self.answer_options,
        )


class QuestionTemplateRegistry:
    """
    Registry of all question templates.
    
    All templates are deterministic and predefined.
    """
    
    def __init__(self):
        self.templates: List[QuestionTemplate] = []
        self.templates_by_id: Dict[str, QuestionTemplate] = {}
        self._register_all_templates()
    
    def _register_all_templates(self) -> None:
        """Register all question templates."""
        
        # -----------------------------------------------------------------
        # DEPLOYMENT QUESTIONS
        # -----------------------------------------------------------------
        
        template = QuestionTemplate(
            template_id="DEP_001",
            question_text="Was there a recent deployment (application or database schema change) in the last 24 hours?",
            evidence_type=EvidenceType.DEPLOYMENT_SCHEMA,
            category=QuestionCategory.DEPLOYMENT,
            priority=QuestionPriority.HIGH,
            expected_answer_type="yes_no_with_time",
            answer_options=[
                "Yes, within last 6 hours",
                "Yes, within last 24 hours",
                "No deployment in last 24 hours",
                "Unknown",
            ],
            confidence_impact=0.15,
        )
        self.templates.append(template)
        self.templates_by_id[template.template_id] = template
        
        template = QuestionTemplate(
            template_id="DEP_002",
            question_text="Were any schema changes made (new columns, indexes, or table modifications)?",
            evidence_type=EvidenceType.DEPLOYMENT_SCHEMA,
            category=QuestionCategory.DEPLOYMENT,
            priority=QuestionPriority.HIGH,
            expected_answer_type="yes_no_description",
            answer_options=[
                "Yes - new column added",
                "Yes - new index created",
                "Yes - table modified",
                "Yes - other change",
                "No schema changes",
            ],
            confidence_impact=0.12,
        )
        self.templates.append(template)
        self.templates_by_id[template.template_id] = template
        
        template = QuestionTemplate(
            template_id="DEP_003",
            question_text="Was a new query or query pattern introduced in the recent deployment?",
            evidence_type=EvidenceType.QUERY_METRICS,
            category=QuestionCategory.DEPLOYMENT,
            priority=QuestionPriority.CRITICAL,
            expected_answer_type="yes_no_with_query",
            answer_options=[
                "Yes - new SELECT query",
                "Yes - new JOIN pattern",
                "Yes - new aggregate function",
                "No new queries",
            ],
            confidence_impact=0.18,
        )
        self.templates.append(template)
        self.templates_by_id[template.template_id] = template
        
        # -----------------------------------------------------------------
        # PERFORMANCE BASELINE QUESTIONS
        # -----------------------------------------------------------------
        
        template = QuestionTemplate(
            template_id="PERF_001",
            question_text="When was the query or incident last performing normally?",
            evidence_type=EvidenceType.PERFORMANCE_BASELINE,
            category=QuestionCategory.PERFORMANCE,
            priority=QuestionPriority.CRITICAL,
            expected_answer_type="timestamp",
            answer_options=[
                "Within last hour",
                "Within last 6 hours",
                "Within last 24 hours",
                "1-7 days ago",
                "Longer than 7 days",
                "Never performed well",
            ],
            confidence_impact=0.20,
        )
        self.templates.append(template)
        self.templates_by_id[template.template_id] = template
        
        template = QuestionTemplate(
            template_id="PERF_002",
            question_text="What was the typical query execution time before the incident?",
            evidence_type=EvidenceType.PERFORMANCE_BASELINE,
            category=QuestionCategory.PERFORMANCE,
            priority=QuestionPriority.HIGH,
            expected_answer_type="duration",
            answer_options=[
                "Under 100ms",
                "100ms - 500ms",
                "500ms - 1s",
                "1s - 5s",
                "Over 5s",
            ],
            confidence_impact=0.15,
        )
        self.templates.append(template)
        self.templates_by_id[template.template_id] = template
        
        template = QuestionTemplate(
            template_id="PERF_003",
            question_text="Did the performance degradation happen suddenly or gradually?",
            evidence_type=EvidenceType.PERFORMANCE_BASELINE,
            category=QuestionCategory.PERFORMANCE,
            priority=QuestionPriority.MEDIUM,
            expected_answer_type="pattern",
            answer_options=[
                "Sudden (minutes)",
                "Gradual (hours)",
                "Gradual (days)",
                "Intermittent",
            ],
            confidence_impact=0.10,
        )
        self.templates.append(template)
        self.templates_by_id[template.template_id] = template
        
        # -----------------------------------------------------------------
        # DATA OPERATIONS QUESTIONS
        # -----------------------------------------------------------------
        
        template = QuestionTemplate(
            template_id="DATA_001",
            question_text="Were any bulk loads or batch jobs running during the incident?",
            evidence_type=EvidenceType.BACKGROUND_JOBS,
            category=QuestionCategory.DATA_OPERATIONS,
            priority=QuestionPriority.HIGH,
            expected_answer_type="yes_no_with_type",
            answer_options=[
                "Yes - bulk INSERT",
                "Yes - bulk UPDATE",
                "Yes - bulk DELETE",
                "Yes - data import",
                "Yes - ETL job",
                "No bulk operations",
            ],
            confidence_impact=0.15,
        )
        self.templates.append(template)
        self.templates_by_id[template.template_id] = template
        
        template = QuestionTemplate(
            template_id="DATA_002",
            question_text="Was there a large data backfill or migration operation?",
            evidence_type=EvidenceType.BACKGROUND_JOBS,
            category=QuestionCategory.DATA_OPERATIONS,
            priority=QuestionPriority.HIGH,
            expected_answer_type="yes_no_scale",
            answer_options=[
                "Yes - over 1M rows",
                "Yes - 100K to 1M rows",
                "Yes - under 100K rows",
                "No data migration",
            ],
            confidence_impact=0.12,
        )
        self.templates.append(template)
        self.templates_by_id[template.template_id] = template
        
        template = QuestionTemplate(
            template_id="DATA_003",
            question_text="Were there any DELETE operations on old data (data cleanup)?",
            evidence_type=EvidenceType.BACKGROUND_JOBS,
            category=QuestionCategory.DATA_OPERATIONS,
            priority=QuestionPriority.MEDIUM,
            expected_answer_type="yes_no_volume",
            answer_options=[
                "Yes - deleted over 1M rows",
                "Yes - deleted 100K to 1M rows",
                "Yes - deleted under 100K rows",
                "No DELETE operations",
            ],
            confidence_impact=0.10,
        )
        self.templates.append(template)
        self.templates_by_id[template.template_id] = template
        
        # -----------------------------------------------------------------
        # MAINTENANCE QUESTIONS
        # -----------------------------------------------------------------
        
        template = QuestionTemplate(
            template_id="MAINT_001",
            question_text="Was autovacuum disabled or paused during the incident?",
            evidence_type=EvidenceType.STATISTICS_MAINTENANCE,
            category=QuestionCategory.MAINTENANCE,
            priority=QuestionPriority.CRITICAL,
            expected_answer_type="yes_no_duration",
            answer_options=[
                "Yes - disabled entirely",
                "Yes - paused for maintenance",
                "Yes - reduced frequency",
                "No - running normally",
                "Unknown",
            ],
            confidence_impact=0.18,
        )
        self.templates.append(template)
        self.templates_by_id[template.template_id] = template
        
        template = QuestionTemplate(
            template_id="MAINT_002",
            question_text="When was the last VACUUM or ANALYZE performed on the affected table?",
            evidence_type=EvidenceType.STATISTICS_MAINTENANCE,
            category=QuestionCategory.MAINTENANCE,
            priority=QuestionPriority.HIGH,
            expected_answer_type="duration_ago",
            answer_options=[
                "Within last 6 hours",
                "Within last 24 hours",
                "1-7 days ago",
                "Over 7 days ago",
                "Never",
            ],
            confidence_impact=0.15,
        )
        self.templates.append(template)
        self.templates_by_id[template.template_id] = template
        
        template = QuestionTemplate(
            template_id="MAINT_003",
            question_text="Were there any scheduled maintenance jobs running (reindex, pg_dump, etc.)?",
            evidence_type=EvidenceType.BACKGROUND_JOBS,
            category=QuestionCategory.MAINTENANCE,
            priority=QuestionPriority.HIGH,
            expected_answer_type="maintenance_type",
            answer_options=[
                "Yes - REINDEX",
                "Yes - pg_dump/pg_restore",
                "Yes - pg_repack",
                "Yes - other maintenance",
                "No maintenance jobs",
            ],
            confidence_impact=0.12,
        )
        self.templates.append(template)
        self.templates_by_id[template.template_id] = template
        
        # -----------------------------------------------------------------
        # CONFIGURATION QUESTIONS
        # -----------------------------------------------------------------
        
        template = QuestionTemplate(
            template_id="CONF_001",
            question_text="Were any PostgreSQL configuration parameters changed recently?",
            evidence_type=EvidenceType.CONFIGURATION,
            category=QuestionCategory.CONFIGURATION,
            priority=QuestionPriority.CRITICAL,
            expected_answer_type="yes_no_with_param",
            answer_options=[
                "Yes - shared_buffers",
                "Yes - work_mem",
                "Yes - effective_cache_size",
                "Yes - other parameter",
                "No parameter changes",
            ],
            confidence_impact=0.18,
        )
        self.templates.append(template)
        self.templates_by_id[template.template_id] = template
        
        template = QuestionTemplate(
            template_id="CONF_002",
            question_text="Were connection pool settings changed (max_connections, pool_size)?",
            evidence_type=EvidenceType.CONFIGURATION,
            category=QuestionCategory.CONFIGURATION,
            priority=QuestionPriority.HIGH,
            expected_answer_type="yes_no_connection",
            answer_options=[
                "Yes - max_connections changed",
                "Yes - pool_size changed",
                "Yes - timeout settings changed",
                "No connection changes",
            ],
            confidence_impact=0.12,
        )
        self.templates.append(template)
        self.templates_by_id[template.template_id] = template
        
        template = QuestionTemplate(
            template_id="CONF_003",
            question_text="Were there any changes to query optimizer settings?",
            evidence_type=EvidenceType.CONFIGURATION,
            category=QuestionCategory.CONFIGURATION,
            priority=QuestionPriority.MEDIUM,
            expected_answer_type="yes_no_optimizer",
            answer_options=[
                "Yes - random_page_cost",
                "Yes - effective_io_concurrency",
                "Yes - planner settings",
                "No optimizer changes",
            ],
            confidence_impact=0.10,
        )
        self.templates.append(template)
        self.templates_by_id[template.template_id] = template
        
        template = QuestionTemplate(
            template_id="CONF_004",
            question_text="Was there a PostgreSQL version upgrade or patch application?",
            evidence_type=EvidenceType.CONFIGURATION,
            category=QuestionCategory.CONFIGURATION,
            priority=QuestionPriority.CRITICAL,
            expected_answer_type="version_change",
            answer_options=[
                "Yes - major version upgrade",
                "Yes - minor version update",
                "Yes - patch applied",
                "No version change",
            ],
            confidence_impact=0.20,
        )
        self.templates.append(template)
        self.templates_by_id[template.template_id] = template
    
    def get_templates_by_evidence_type(
        self,
        evidence_type: EvidenceType,
    ) -> List[QuestionTemplate]:
        """Get all templates for a specific evidence type."""
        return [t for t in self.templates if t.evidence_type == evidence_type]
    
    def get_templates_by_category(
        self,
        category: QuestionCategory,
    ) -> List[QuestionTemplate]:
        """Get all templates for a specific category."""
        return [t for t in self.templates if t.category == category]
    
    def get_highest_priority_template(
        self,
        evidence_type: EvidenceType,
    ) -> Optional[QuestionTemplate]:
        """Get the highest priority template for an evidence type."""
        templates = self.get_templates_by_evidence_type(evidence_type)
        if not templates:
            return None
        return min(templates, key=lambda t: t.priority.value)


# ---------------------------------------------------------------------
# Evidence Gap Analyzer
# ---------------------------------------------------------------------

@dataclass
class EvidenceGap:
    """A gap in evidence that can be filled by asking questions."""
    evidence_type: EvidenceType
    missing_count: int
    is_required: bool
    current_confidence_impact: float
    potential_confidence_impact: float
    related_signals: List[str] = field(default_factory=list)


class EvidenceGapAnalyzer:
    """
    Analyzes evidence collection to identify gaps.
    
    Compares expected evidence against collected evidence to find:
    - Missing required evidence
    - Missing optional evidence
    - Evidence that could improve confidence
    """
    
    # Mapping of root cause categories to evidence types
    EVIDENCE_TYPE_MAPPING = {
        "INDEX_ISSUES": [EvidenceType.INDEX_HEALTH, EvidenceType.QUERY_METRICS],
        "STATISTICS_MAINTENANCE": [EvidenceType.TABLE_STATS, EvidenceType.STATISTICS_MAINTENANCE],
        "BLOCKING_LOCKING": [EvidenceType.LOCKING],
        "CONFIGURATION": [EvidenceType.CONFIGURATION],
        "APPLICATION_BEHAVIOR": [EvidenceType.QUERY_METRICS],
        "CAPACITY_HARDWARE": [EvidenceType.HARDWARE_CAPACITY, EvidenceType.QUERY_METRICS],
        "BACKGROUND_JOBS": [EvidenceType.BACKGROUND_JOBS],
        "DEPLOYMENT_SCHEMA": [EvidenceType.DEPLOYMENT_SCHEMA],
    }
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
    
    def analyze_gaps(
        self,
        root_cause_results: Dict[str, Any],
        evidence_collection: Dict[str, Any],
    ) -> List[EvidenceGap]:
        """
        Analyze evidence gaps from root cause analysis.
        
        Args:
            root_cause_results: Results from RootCauseEngine
            evidence_collection: Evidence collection from EvidenceBuilder
            
        Returns:
            List of EvidenceGap objects sorted by potential confidence impact
        """
        self.logger.info("Analyzing evidence gaps")
        
        gaps: List[EvidenceGap] = []
        
        # Get evidence types present
        present_types = self._get_present_evidence_types(evidence_collection)
        
        # Analyze each likely root cause
        likely_causes = root_cause_results.get("likely_causes", [])
        for cause in likely_causes:
            cause_category = cause.get("category", "")
            expected_types = self.EVIDENCE_TYPE_MAPPING.get(cause_category, [])
            
            for evidence_type in expected_types:
                if evidence_type not in present_types:
                    gap = EvidenceGap(
                        evidence_type=evidence_type,
                        missing_count=1,
                        is_required=True,
                        current_confidence_impact=0.0,
                        potential_confidence_impact=self._estimate_confidence_impact(
                            evidence_type, cause_category
                        ),
                        related_signals=[cause.get("cause_id", "")],
                    )
                    gaps.append(gap)
        
        # Sort by potential impact (highest first)
        gaps.sort(key=lambda g: g.potential_confidence_impact, reverse=True)
        
        self.logger.info("Found %d evidence gaps", len(gaps))
        return gaps
    
    def _get_present_evidence_types(
        self,
        evidence_collection: Dict[str, Any],
    ) -> set:
        """Get evidence types that are already present."""
        present = set()
        
        evidence = evidence_collection.get("evidence", [])
        for ev in evidence:
            ev_type = ev.get("evidence_type", "").lower()
            # Map evidence_type string to EvidenceType enum
            for et in EvidenceType:
                if et.value in ev_type:
                    present.add(et)
                    break
        
        return present
    
    def _estimate_confidence_impact(
        self,
        evidence_type: EvidenceType,
        cause_category: str,
    ) -> float:
        """Estimate how much confidence would improve with this evidence."""
        # Base impacts by evidence type
        base_impacts = {
            EvidenceType.DEPLOYMENT_SCHEMA: 0.15,
            EvidenceType.PERFORMANCE_BASELINE: 0.20,
            EvidenceType.BACKGROUND_JOBS: 0.12,
            EvidenceType.STATISTICS_MAINTENANCE: 0.15,
            EvidenceType.CONFIGURATION: 0.18,
            EvidenceType.QUERY_METRICS: 0.10,
            EvidenceType.TABLE_STATS: 0.10,
            EvidenceType.INDEX_HEALTH: 0.10,
            EvidenceType.LOCKING: 0.12,
            EvidenceType.HARDWARE_CAPACITY: 0.10,
        }
        
        base = base_impacts.get(evidence_type, 0.10)
        
        # Adjust based on cause category priority
        high_priority_categories = ["DEPLOYMENT_SCHEMA", "CONFIGURATION", "STATISTICS_MAINTENANCE"]
        if cause_category in high_priority_categories:
            base *= 1.2
        
        return min(base, 0.25)  # Cap at 25% per evidence type


# ---------------------------------------------------------------------
# Clarification Manager
# ---------------------------------------------------------------------

class ClarificationManager:
    """
    Main orchestrator for the clarification question framework.
    
    Responsibilities:
    1. Determine when to ask questions (decision logic)
    2. Generate appropriate questions from templates
    3. Track clarification state
    4. Process answers and update confidence
    """
    
    # Confidence thresholds
    CONFIDENCE_THRESHOLD_HIGH = 0.85  # No questions needed
    CONFIDENCE_THRESHOLD_MODERATE = 0.75  # Questions beneficial
    CONFIDENCE_THRESHOLD_LOW = 0.60  # Questions strongly recommended
    
    def __init__(
        self,
        confidence_threshold: float = 0.75,
        max_questions: int = 5,
        log_level: str = "INFO",
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        
        self.confidence_threshold = confidence_threshold
        self.max_questions = max_questions
        self.template_registry = QuestionTemplateRegistry()
        self.gap_analyzer = EvidenceGapAnalyzer()
        
        # State
        self.current_state: Optional[ClarificationState] = None
    
    def should_ask_question(
        self,
        confidence_score: float,
        evidence_gaps: List[EvidenceGap],
    ) -> bool:
        """
        Determine if a clarification question should be asked.
        
        Decision logic:
        1. Confidence < threshold
        2. Evidence gaps exist
        3. Haven't exceeded max questions
        4. Not already asked similar question
        """
        # Check confidence threshold
        if confidence_score >= self.confidence_threshold:
            self.logger.debug("Confidence %f >= threshold %f - no questions needed",
                            confidence_score, self.confidence_threshold)
            return False
        
        # Check if we have gaps to fill
        if not evidence_gaps:
            self.logger.debug("No evidence gaps - no questions needed")
            return False
        
        # Check max questions
        if self.current_state and self.current_state.questions_asked_count >= self.max_questions:
            self.logger.debug("Max questions reached (%d)", self.max_questions)
            return False
        
        return True
    
    def generate_question(
        self,
        evidence_gaps: List[EvidenceGap],
        confidence_score: float,
    ) -> Optional[ClarificationQuestion]:
        """
        Generate the next clarification question.
        
        Strategy:
        1. Pick highest impact gap first
        2. Get highest priority template for that gap
        3. Generate question from template
        """
        if not evidence_gaps:
            return None
        
        # Pick the gap with highest potential confidence impact
        highest_gap = evidence_gaps[0]
        
        # Get template for this evidence type
        template = self.template_registry.get_highest_priority_template(
            highest_gap.evidence_type
        )
        
        if not template:
            self.logger.warning("No template found for evidence type: %s",
                              highest_gap.evidence_type)
            return None
        
        # Generate question
        question = template.generate_question(
            missing_evidence_count=highest_gap.missing_count,
            confidence_threshold=self.confidence_threshold,
        )
        
        self.logger.info("Generated question: %s", question.question_text)
        return question
    
    def initialize_state(
        self,
        confidence_score: float,
        evidence_gaps: List[EvidenceGap],
    ) -> ClarificationState:
        """Initialize clarification state for an incident."""
        self.current_state = ClarificationState(
            original_confidence=confidence_score,
            current_confidence=confidence_score,
            target_confidence=self.confidence_threshold,
            max_questions=self.max_questions,
        )
        return self.current_state
    
    def process_answer(
        self,
        question: ClarificationQuestion,
        answer_value: Any,
    ) -> ClarificationAnswer:
        """
        Process an answer to a clarification question.
        
        Returns a structured answer with confidence impact.
        """
        # Determine confidence impact based on answer
        confidence_impact = self._calculate_confidence_impact(
            question, answer_value
        )
        
        answer = ClarificationAnswer(
            question_id=question.question_id,
            category=question.category,
            answer_value=answer_value,
            confidence_impact=confidence_impact,
            evidence_generated=self._generate_evidence_ids(question, answer_value),
        )
        
        # Update state
        if self.current_state:
            self.current_state.answers_received.append(answer)
            self.current_state.current_confidence = min(
                self.current_state.current_confidence + confidence_impact,
                1.0
            )
            self.current_state.questions_asked_count += 1
        
        self.logger.info("Processed answer: confidence impact = %f", confidence_impact)
        return answer
    
    def _calculate_confidence_impact(
        self,
        question: ClarificationQuestion,
        answer_value: Any,
    ) -> float:
        """Calculate how much confidence improves based on answer."""
        # Get template for this question
        template = self.template_registry.templates_by_id.get(
            question.template_id, None
        )
        
        if not template:
            return question.confidence_threshold * 0.1
        
        base_impact = template.confidence_impact
        
        # Adjust based on answer quality
        if answer_value in ["Unknown", "No", "No changes", "No deployment in last 24 hours"]:
            # Answer provides negative information - lower impact
            return base_impact * 0.5
        elif answer_value in ["Yes - within last 6 hours", "Sudden (minutes)"]:
            # Answer provides specific timing - full impact
            return base_impact
        else:
            # General affirmative - moderate impact
            return base_impact * 0.8
    
    def _generate_evidence_ids(
        self,
        question: ClarificationQuestion,
        answer_value: Any,
    ) -> List[str]:
        """Generate evidence IDs that this answer would create."""
        evidence_ids = []
        
        if answer_value not in ["Unknown", "No", "No changes"]:
            evidence_ids.append(f"ev_{question.evidence_type.value}_{question.question_id}")
        
        return evidence_ids
    
    def get_next_question(
        self,
        confidence_score: float,
        root_cause_results: Dict[str, Any],
        evidence_collection: Dict[str, Any],
    ) -> Optional[ClarificationQuestion]:
        """
        Get the next clarification question.
        
        This is the main entry point for generating questions.
        """
        # Initialize state if not done
        if not self.current_state:
            evidence_gaps = self.gap_analyzer.analyze_gaps(
                root_cause_results, evidence_collection
            )
            self.initialize_state(confidence_score, evidence_gaps)
            # Use the analyzed gaps for question generation
            gaps_for_generation = evidence_gaps
        else:
            gaps_for_generation = self.gap_analyzer.analyze_gaps(
                root_cause_results, evidence_collection
            )
        
        # Check if we should ask
        if not self.should_ask_question(
            confidence_score,
            gaps_for_generation
        ):
            return None
        
        # Generate next question
        question = self.generate_question(
            gaps_for_generation,
            confidence_score
        )
        
        if question and self.current_state:
            self.current_state.questions_asked.append(question)
            self.current_state.pending_questions.append(question)
        
        return question
    
    def get_state(self) -> Optional[ClarificationState]:
        """Get current clarification state."""
        return self.current_state
    
    def reset(self) -> None:
        """Reset clarification state for new incident."""
        self.current_state = None
        self.logger.info("Clarification state reset")


# ---------------------------------------------------------------------
# Integration Helper Functions
# ---------------------------------------------------------------------

def create_clarification_from_confidence(
    confidence_breakdown: Dict[str, Any],
    root_cause_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create clarification questions based on confidence breakdown.
    
    Integration helper for ConfidenceScorer output.
    
    Args:
        confidence_breakdown: Output from ConfidenceScorer.explain_score()
        root_cause_results: Output from RootCauseEngine
        
    Returns:
        Dictionary with questions and state
    """
    manager = ClarificationManager()
    
    score = confidence_breakdown.get("score", 0.0)
    
    # Build minimal evidence collection
    evidence_collection = {
        "evidence": confidence_breakdown.get("breakdown", {}).get("evidence", [])
    }
    
    # Get next question
    question = manager.get_next_question(
        score,
        root_cause_results,
        evidence_collection,
    )
    
    return {
        "should_ask": question is not None,
        "question": question.to_dict() if question else None,
        "state": manager.get_state().to_dict() if manager.get_state() else None,
        "score": score,
        "rating": confidence_breakdown.get("rating", "Unknown"),
    }


def get_clarification_questions_for_gap(
    evidence_type: EvidenceType,
) -> List[ClarificationQuestion]:
    """Get all possible questions for a specific evidence gap."""
    registry = QuestionTemplateRegistry()
    templates = registry.get_templates_by_evidence_type(evidence_type)
    
    return [
        template.generate_question()
        for template in templates
    ]


# ---------------------------------------------------------------------
# Main execution for testing
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Clarification-Question Framework - Example Usage")
    print("=" * 70)
    
    # Example 1: Generate a question
    manager = ClarificationManager(confidence_threshold=0.75, max_questions=5)
    
    print("\n--- Example 1: Question Generation ---")
    
    # Simulate evidence gaps
    gaps = [
        EvidenceGap(
            evidence_type=EvidenceType.DEPLOYMENT_SCHEMA,
            missing_count=1,
            is_required=True,
            current_confidence_impact=0.0,
            potential_confidence_impact=0.15,
        ),
        EvidenceGap(
            evidence_type=EvidenceType.STATISTICS_MAINTENANCE,
            missing_count=1,
            is_required=True,
            current_confidence_impact=0.0,
            potential_confidence_impact=0.12,
        ),
    ]
    
    question = manager.generate_question(gaps, 0.65)
    if question:
        print(f"Generated Question:")
        print(f"  ID: {question.question_id}")
        print(f"  Text: {question.question_text}")
        print(f"  Category: {question.category.value}")
        print(f"  Priority: {question.priority.name}")
        print(f"  Options: {', '.join(question.answer_options[:3])}...")
    
    # Example 2: Process an answer
    print("\n--- Example 2: Processing Answer ---")
    
    if question:
        answer = manager.process_answer(question, "Yes, within last 6 hours")
        print(f"Processed Answer:")
        print(f"  Answer Value: {answer.answer_value}")
        print(f"  Confidence Impact: {answer.confidence_impact:.2%}")
        print(f"  Evidence Generated: {answer.evidence_generated}")
    
    # Example 3: State tracking
    print("\n--- Example 3: State Tracking ---")
    
    state = manager.get_state()
    if state:
        print(f"State:")
        print(f"  Original Confidence: {state.original_confidence:.2%}")
        print(f"  Current Confidence: {state.current_confidence:.2%}")
        print(f"  Questions Asked: {state.questions_asked_count}")
        print(f"  Target Confidence: {state.target_confidence:.2%}")
    
    # Example 4: List all templates
    print("\n--- Example 4: Available Question Templates ---")
    
    registry = QuestionTemplateRegistry()
    print(f"Total templates registered: {len(registry.templates)}")
    
    for category in QuestionCategory:
        category_templates = registry.get_templates_by_category(category)
        print(f"  {category.value}: {len(category_templates)} templates")
    
    # Example 5: Evidence gap analysis
    print("\n--- Example 5: Evidence Gap Analysis ---")
    
    analyzer = EvidenceGapAnalyzer()
    sample_root_cause = {
        "likely_causes": [
            {"category": "DEPLOYMENT_SCHEMA", "cause_id": "rc_001"},
            {"category": "STATISTICS_MAINTENANCE", "cause_id": "rc_002"},
        ]
    }
    sample_evidence = {"evidence": []}
    
    gaps = analyzer.analyze_gaps(sample_root_cause, sample_evidence)
    print(f"Identified {len(gaps)} evidence gaps:")
    for gap in gaps:
        print(f"  - {gap.evidence_type.value}: potential impact = {gap.potential_confidence_impact:.2%}")

