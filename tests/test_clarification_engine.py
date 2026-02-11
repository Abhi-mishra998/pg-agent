#!/usr/bin/env python3
"""
Unit Tests for Clarification-Question Framework

Tests cover:
1. QuestionTemplateRegistry - template registration and lookup
2. EvidenceGapAnalyzer - evidence gap identification
3. ClarificationManager - decision logic and question generation
4. Integration - end-to-end clarification flow
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.clarification_engine import (
    EvidenceType,
    QuestionPriority,
    QuestionCategory,
    ClarificationAnswer,
    ClarificationQuestion,
    ClarificationState,
    QuestionTemplate,
    QuestionTemplateRegistry,
    EvidenceGap,
    EvidenceGapAnalyzer,
    ClarificationManager,
    create_clarification_from_confidence,
    get_clarification_questions_for_gap,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def registry():
    """Create a fresh question template registry for each test."""
    return QuestionTemplateRegistry()


@pytest.fixture
def analyzer():
    """Create a fresh evidence gap analyzer for each test."""
    return EvidenceGapAnalyzer()


@pytest.fixture
def manager():
    """Create a fresh clarification manager for each test."""
    return ClarificationManager(confidence_threshold=0.75, max_questions=5)


@pytest.fixture
def sample_root_cause_results():
    """Sample root cause analysis results."""
    return {
        "likely_causes": [
            {"category": "DEPLOYMENT_SCHEMA", "cause_id": "rc_001"},
            {"category": "STATISTICS_MAINTENANCE", "cause_id": "rc_002"},
        ]
    }


@pytest.fixture
def sample_evidence_collection():
    """Sample evidence collection with some evidence present."""
    return {
        "evidence": [
            {"evidence_type": "query_metrics", "description": "Query analysis"},
            {"evidence_type": "table_stats", "description": "Table statistics"},
        ]
    }


# ---------------------------------------------------------------------
# Tests for QuestionTemplateRegistry
# ---------------------------------------------------------------------

class TestQuestionTemplateRegistry:
    """Tests for the question template registry."""
    
    def test_registry_has_templates(self, registry):
        """Verify that templates are registered."""
        assert len(registry.templates) > 0
        assert len(registry.templates_by_id) > 0
    
    def test_all_templates_have_id(self, registry):
        """Verify all templates have unique IDs."""
        for template in registry.templates:
            assert template.template_id is not None
            assert len(template.template_id) > 0
    
    def test_templates_by_id_lookup(self, registry):
        """Verify templates can be looked up by ID."""
        for template in registry.templates:
            looked_up = registry.templates_by_id.get(template.template_id)
            assert looked_up is not None
            assert looked_up.template_id == template.template_id
    
    def test_get_templates_by_evidence_type(self, registry):
        """Verify filtering templates by evidence type."""
        # Test CONFIGURATION type
        config_templates = registry.get_templates_by_evidence_type(
            EvidenceType.CONFIGURATION
        )
        assert len(config_templates) > 0
        for t in config_templates:
            assert t.evidence_type == EvidenceType.CONFIGURATION
    
    def test_get_templates_by_category(self, registry):
        """Verify filtering templates by category."""
        # Test DEPLOYMENT category
        deployment_templates = registry.get_templates_by_category(
            QuestionCategory.DEPLOYMENT
        )
        assert len(deployment_templates) > 0
        for t in deployment_templates:
            assert t.category == QuestionCategory.DEPLOYMENT
    
    def test_get_highest_priority_template(self, registry):
        """Verify getting highest priority template for evidence type."""
        template = registry.get_highest_priority_template(
            EvidenceType.CONFIGURATION
        )
        assert template is not None
        # Verify it's the highest priority (lowest value)
        all_templates = registry.get_templates_by_evidence_type(
            EvidenceType.CONFIGURATION
        )
        # Compare priority values (QuestionPriority enum values are integers)
        min_priority_value = min(t.priority.value for t in all_templates)
        assert template.priority.value == min_priority_value
    
    def test_template_generates_question(self, registry):
        """Verify template can generate a question."""
        template = registry.templates[0]
        question = template.generate_question(
            missing_evidence_count=2,
            confidence_threshold=0.70
        )
        
        assert isinstance(question, ClarificationQuestion)
        assert question.template_id == template.template_id
        assert question.question_text == template.question_text
        assert question.category == template.category
        assert question.evidence_type == template.evidence_type
        assert question.missing_evidence_count == 2
        assert question.confidence_threshold == 0.70
        assert len(question.answer_options) > 0
    
    def test_all_categories_have_templates(self, registry):
        """Verify all question categories have at least one template."""
        for category in QuestionCategory:
            templates = registry.get_templates_by_category(category)
            assert len(templates) > 0, f"Category {category} has no templates"
    
    def test_all_templates_have_answer_options(self, registry):
        """Verify all templates have structured answer options."""
        for template in registry.templates:
            assert len(template.answer_options) > 0
            # Options should be structured, not free-text
            assert len(template.answer_options) <= 6  # Reasonable limit


# ---------------------------------------------------------------------
# Tests for EvidenceGapAnalyzer
# ---------------------------------------------------------------------

class TestEvidenceGapAnalyzer:
    """Tests for the evidence gap analyzer."""
    
    def test_analyze_gaps_empty_collection(self, analyzer, sample_root_cause_results):
        """Test gap analysis with empty evidence collection."""
        empty_collection = {"evidence": []}
        gaps = analyzer.analyze_gaps(sample_root_cause_results, empty_collection)
        
        assert len(gaps) > 0
        # Should find gaps for DEPLOYMENT_SCHEMA and STATISTICS_MAINTENANCE
        gap_types = [g.evidence_type for g in gaps]
        assert EvidenceType.DEPLOYMENT_SCHEMA in gap_types
        assert EvidenceType.STATISTICS_MAINTENANCE in gap_types
    
    def test_analyze_gaps_with_evidence(self, analyzer, sample_root_cause_results):
        """Test gap analysis with some evidence already present."""
        evidence_collection = {
            "evidence": [
                {"evidence_type": "query_metrics", "description": "Query analysis"},
                {"evidence_type": "table_stats", "description": "Table statistics"},
            ]
        }
        gaps = analyzer.analyze_gaps(sample_root_cause_results, evidence_collection)
        
        # Should have fewer gaps since some evidence exists
        # Note: The matching is substring-based, so this is a basic sanity check
        assert isinstance(gaps, list)
    
    def test_gaps_sorted_by_impact(self, analyzer, sample_root_cause_results):
        """Verify gaps are sorted by potential confidence impact."""
        empty_collection = {"evidence": []}
        gaps = analyzer.analyze_gaps(sample_root_cause_results, empty_collection)
        
        # Should be sorted descending by potential_impact
        impacts = [g.potential_confidence_impact for g in gaps]
        assert impacts == sorted(impacts, reverse=True)
    
    def test_estimate_confidence_impact(self, analyzer):
        """Verify confidence impact estimation."""
        # High priority evidence types should have higher impact
        perf_impact = analyzer._estimate_confidence_impact(
            EvidenceType.PERFORMANCE_BASELINE, "DEPLOYMENT_SCHEMA"
        )
        config_impact = analyzer._estimate_confidence_impact(
            EvidenceType.CONFIGURATION, "DEPLOYMENT_SCHEMA"
        )
        
        # Both should be > 0
        assert perf_impact > 0
        assert config_impact > 0
    
    def test_estimate_confidence_impact_capped(self, analyzer):
        """Verify confidence impact is capped at 25%."""
        # Even with multiplier, should be capped
        impact = analyzer._estimate_confidence_impact(
            EvidenceType.PERFORMANCE_BASELINE, "DEPLOYMENT_SCHEMA"
        )
        assert impact <= 0.25
    
    def test_get_present_evidence_types(self, analyzer):
        """Verify extraction of present evidence types."""
        collection = {
            "evidence": [
                {"evidence_type": "query_metrics"},
                {"evidence_type": "TABLE_STATS"},
            ]
        }
        present = analyzer._get_present_evidence_types(collection)
        
        assert EvidenceType.QUERY_METRICS in present
        assert EvidenceType.TABLE_STATS in present


# ---------------------------------------------------------------------
# Tests for ClarificationManager
# ---------------------------------------------------------------------

class TestClarificationManager:
    """Tests for the clarification manager."""
    
    def test_should_ask_question_high_confidence(self, manager):
        """Verify no question when confidence is high."""
        assert manager.should_ask_question(0.90, []) is False
        assert manager.should_ask_question(0.85, []) is False
    
    def test_should_ask_question_at_threshold(self, manager):
        """Verify no question when confidence equals threshold."""
        assert manager.should_ask_question(0.75, []) is False
    
    def test_should_ask_question_low_confidence_no_gaps(self, manager):
        """Verify no question when confidence is low but no gaps."""
        assert manager.should_ask_question(0.65, []) is False
    
    def test_should_ask_question_low_confidence_with_gaps(self, manager):
        """Verify question is asked when confidence is low and gaps exist."""
        gaps = [
            EvidenceGap(
                evidence_type=EvidenceType.DEPLOYMENT_SCHEMA,
                missing_count=1,
                is_required=True,
                current_confidence_impact=0.0,
                potential_confidence_impact=0.15,
            )
        ]
        assert manager.should_ask_question(0.65, gaps) is True
    
    def test_max_questions_limit(self, manager):
        """Verify max questions limit is enforced."""
        # Initialize state
        gaps = [EvidenceGap(
            evidence_type=EvidenceType.DEPLOYMENT_SCHEMA,
            missing_count=1,
            is_required=True,
            current_confidence_impact=0.0,
            potential_confidence_impact=0.15,
        )]
        manager.initialize_state(0.65, gaps)
        
        # Simulate asking max questions
        manager.current_state.questions_asked_count = 5
        
        assert manager.should_ask_question(0.65, gaps) is False
    
    def test_generate_question(self, manager):
        """Verify question generation from evidence gaps."""
        gaps = [
            EvidenceGap(
                evidence_type=EvidenceType.DEPLOYMENT_SCHEMA,
                missing_count=1,
                is_required=True,
                current_confidence_impact=0.0,
                potential_confidence_impact=0.15,
            )
        ]
        question = manager.generate_question(gaps, 0.65)
        
        assert question is not None
        assert isinstance(question, ClarificationQuestion)
        assert question.evidence_type == EvidenceType.DEPLOYMENT_SCHEMA
        assert question.confidence_threshold == 0.75
    
    def test_generate_question_no_gaps(self, manager):
        """Verify no question when no gaps."""
        question = manager.generate_question([], 0.65)
        assert question is None
    
    def test_initialize_state(self, manager):
        """Verify state initialization."""
        gaps = [EvidenceGap(
            evidence_type=EvidenceType.DEPLOYMENT_SCHEMA,
            missing_count=1,
            is_required=True,
            current_confidence_impact=0.0,
            potential_confidence_impact=0.15,
        )]
        state = manager.initialize_state(0.65, gaps)
        
        assert state.original_confidence == 0.65
        assert state.current_confidence == 0.65
        assert state.target_confidence == 0.75
        assert state.max_questions == 5
        assert state.questions_asked_count == 0
    
    def test_process_answer(self, manager):
        """Verify answer processing and confidence update."""
        # Initialize
        gaps = [EvidenceGap(
            evidence_type=EvidenceType.DEPLOYMENT_SCHEMA,
            missing_count=1,
            is_required=True,
            current_confidence_impact=0.0,
            potential_confidence_impact=0.15,
        )]
        manager.initialize_state(0.65, gaps)
        
        # Generate a question
        question = manager.generate_question(gaps, 0.65)
        
        # Process answer
        answer = manager.process_answer(question, "Yes, within last 6 hours")
        
        assert answer is not None
        assert answer.question_id == question.question_id
        assert answer.category == question.category
        assert answer.answer_value == "Yes, within last 6 hours"
        assert answer.confidence_impact > 0
        assert len(answer.evidence_generated) > 0
        
        # Verify state was updated
        assert manager.current_state.current_confidence > 0.65
        assert manager.current_state.questions_asked_count == 1
    
    def test_process_answer_negative(self, manager):
        """Verify confidence impact for negative answers."""
        gaps = [EvidenceGap(
            evidence_type=EvidenceType.DEPLOYMENT_SCHEMA,
            missing_count=1,
            is_required=True,
            current_confidence_impact=0.0,
            potential_confidence_impact=0.15,
        )]
        manager.initialize_state(0.65, gaps)
        question = manager.generate_question(gaps, 0.65)
        
        # Negative answer should have lower impact
        negative_answer = manager.process_answer(question, "No")
        positive_answer = manager.process_answer(question, "Yes, within last 6 hours")
        
        assert negative_answer.confidence_impact < positive_answer.confidence_impact
    
    def test_get_next_question(self, manager, sample_root_cause_results):
        """Verify getting the next question."""
        question = manager.get_next_question(
            0.65,
            sample_root_cause_results,
            {"evidence": []}
        )
        
        assert question is not None
        assert manager.current_state is not None
        # questions_asked_count is incremented when answers are processed
        # questions_asked list contains the actual questions
        assert len(manager.current_state.questions_asked) == 1
    
    def test_get_next_question_high_confidence(self, manager, sample_root_cause_results):
        """Verify no question when confidence is high."""
        question = manager.get_next_question(
            0.90,
            sample_root_cause_results,
            {"evidence": []}
        )
        
        assert question is None
    
    def test_reset(self, manager):
        """Verify state reset."""
        gaps = [EvidenceGap(
            evidence_type=EvidenceType.DEPLOYMENT_SCHEMA,
            missing_count=1,
            is_required=True,
            current_confidence_impact=0.0,
            potential_confidence_impact=0.15,
        )]
        manager.initialize_state(0.65, gaps)
        manager.current_state.questions_asked_count = 3
        
        manager.reset()
        
        assert manager.current_state is None
    
    def test_confidence_never_exceeds_one(self, manager):
        """Verify confidence doesn't exceed 1.0 after multiple answers."""
        gaps = [
            EvidenceGap(
                evidence_type=EvidenceType.DEPLOYMENT_SCHEMA,
                missing_count=1,
                is_required=True,
                current_confidence_impact=0.0,
                potential_confidence_impact=0.15,
            ),
            EvidenceGap(
                evidence_type=EvidenceType.CONFIGURATION,
                missing_count=1,
                is_required=True,
                current_confidence_impact=0.0,
                potential_confidence_impact=0.18,
            ),
        ]
        manager.initialize_state(0.60, gaps)
        
        # Answer multiple questions
        for i in range(5):
            question = manager.generate_question(gaps, manager.current_state.current_confidence)
            if question:
                manager.process_answer(question, "Yes - within last 6 hours")
        
        assert manager.current_state.current_confidence <= 1.0


# ---------------------------------------------------------------------
# Tests for Data Models
# ---------------------------------------------------------------------

class TestDataModels:
    """Tests for data model serialization and methods."""
    
    def test_clarification_question_to_dict(self):
        """Verify ClarificationQuestion serialization."""
        question = ClarificationQuestion(
            question_id="q_test_001",
            question_text="Test question?",
            category=QuestionCategory.DEPLOYMENT,
            evidence_type=EvidenceType.DEPLOYMENT_SCHEMA,
            priority=QuestionPriority.HIGH,
            missing_evidence_count=2,
            confidence_threshold=0.75,
            template_id="DEP_001",
            expected_answer_type="yes_no",
            answer_options=["Yes", "No"],
        )
        
        result = question.to_dict()
        
        assert result["question_id"] == "q_test_001"
        assert result["question_text"] == "Test question?"
        assert result["category"] == "DEPLOYMENT"
        assert result["evidence_type"] == "deployment_schema"
        assert result["priority"] == 2
        assert result["answer_options"] == ["Yes", "No"]
    
    def test_clarification_question_to_human_readable(self):
        """Verify human-readable format."""
        question = ClarificationQuestion(
            question_id="q_test_001",
            question_text="Test question?",
            category=QuestionCategory.DEPLOYMENT,
            evidence_type=EvidenceType.DEPLOYMENT_SCHEMA,
            priority=QuestionPriority.HIGH,
            missing_evidence_count=1,
            confidence_threshold=0.75,
            template_id="DEP_001",
            expected_answer_type="yes_no",
            answer_options=["Yes", "No"],
        )
        
        result = question.to_human_readable()
        
        assert "[DEPLOYMENT]" in result
        assert "Test question?" in result
    
    def test_clarification_answer_to_dict(self):
        """Verify ClarificationAnswer serialization."""
        answer = ClarificationAnswer(
            question_id="q_test_001",
            category=QuestionCategory.DEPLOYMENT,
            answer_value="Yes",
            confidence_impact=0.15,
            evidence_generated=["ev_001"],
        )
        
        result = answer.to_dict()
        
        assert result["question_id"] == "q_test_001"
        assert result["category"] == "DEPLOYMENT"
        assert result["answer_value"] == "Yes"
        assert result["confidence_impact"] == 0.15
        assert result["evidence_generated"] == ["ev_001"]
    
    def test_clarification_state_to_dict(self):
        """Verify ClarificationState serialization."""
        state = ClarificationState(
            original_confidence=0.65,
            current_confidence=0.75,
            target_confidence=0.75,
            max_questions=5,
            questions_asked_count=2,
        )
        
        result = state.to_dict()
        
        assert result["original_confidence"] == 0.65
        assert result["current_confidence"] == 0.75
        assert result["target_confidence"] == 0.75
        assert result["questions_asked_count"] == 2


# ---------------------------------------------------------------------
# Tests for Integration Helper Functions
# ---------------------------------------------------------------------

class TestIntegrationHelpers:
    """Tests for integration helper functions."""
    
    def test_create_clarification_from_confidence_high_score(self):
        """Test with high confidence score."""
        breakdown = {
            "score": 0.90,
            "rating": "High",
            "breakdown": {
                "component_details": {
                    "missing_evidence_count": 0
                },
                "evidence": []
            }
        }
        
        result = create_clarification_from_confidence(
            breakdown,
            {"likely_causes": []}
        )
        
        assert result["should_ask"] is False
        assert result["question"] is None
        assert result["score"] == 0.90
    
    def test_create_clarification_from_confidence_low_score(self):
        """Test with low confidence score."""
        breakdown = {
            "score": 0.60,
            "rating": "Low",
            "breakdown": {
                "component_details": {
                    "missing_evidence_count": 2
                },
                "evidence": []
            }
        }
        
        result = create_clarification_from_confidence(
            breakdown,
            {"likely_causes": [{"category": "DEPLOYMENT_SCHEMA"}]}
        )
        
        assert result["should_ask"] is True
        assert result["question"] is not None
        assert result["score"] == 0.60
    
    def test_get_clarification_questions_for_gap(self):
        """Test getting questions for specific evidence gap."""
        questions = get_clarification_questions_for_gap(
            EvidenceType.CONFIGURATION
        )
        
        assert len(questions) > 0
        for q in questions:
            assert q.evidence_type == EvidenceType.CONFIGURATION


# ---------------------------------------------------------------------
# Tests for Decision Logic
# ---------------------------------------------------------------------

class TestDecisionLogic:
    """Tests for the decision logic of when to ask questions."""
    
    def test_threshold_behavior(self, manager):
        """Test that threshold is properly enforced."""
        gaps = [EvidenceGap(
            evidence_type=EvidenceType.DEPLOYMENT_SCHEMA,
            missing_count=1,
            is_required=True,
            current_confidence_impact=0.0,
            potential_confidence_impact=0.15,
        )]
        
        # At threshold - no question
        assert manager.should_ask_question(0.75, gaps) is False
        
        # Below threshold - ask question
        assert manager.should_ask_question(0.74, gaps) is True
    
    def test_gap_requirement(self, manager):
        """Test that gaps are required to ask questions."""
        # No gaps - no question even at low confidence
        assert manager.should_ask_question(0.50, []) is False
        
        # With gaps - ask question
        gaps = [EvidenceGap(
            evidence_type=EvidenceType.DEPLOYMENT_SCHEMA,
            missing_count=1,
            is_required=True,
            current_confidence_impact=0.0,
            potential_confidence_impact=0.15,
        )]
        assert manager.should_ask_question(0.50, gaps) is True


# ---------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_root_cause_results(self, manager):
        """Test with empty root cause results."""
        question = manager.get_next_question(
            0.65,
            {"likely_causes": []},
            {"evidence": []}
        )
        
        # Should not crash, may or may not return question
        # depending on gap analysis
        assert question is None or isinstance(question, ClarificationQuestion)
    
    def test_unknown_evidence_type(self, registry):
        """Test handling of unknown evidence type."""
        # Should not crash when getting templates for unknown type
        templates = registry.get_templates_by_evidence_type(
            EvidenceType.HARDWARE_CAPACITY
        )
        # May or may not have templates
        assert isinstance(templates, list)
    
    def test_very_low_confidence(self, manager):
        """Test with very low confidence score."""
        gaps = [
            EvidenceGap(
                evidence_type=EvidenceType.DEPLOYMENT_SCHEMA,
                missing_count=1,
                is_required=True,
                current_confidence_impact=0.0,
                potential_confidence_impact=0.15,
            ),
            EvidenceGap(
                evidence_type=EvidenceType.CONFIGURATION,
                missing_count=1,
                is_required=True,
                current_confidence_impact=0.0,
                potential_confidence_impact=0.18,
            ),
        ]
        manager.initialize_state(0.10, gaps)
        
        # Should handle very low confidence
        question = manager.generate_question(gaps, 0.10)
        assert question is not None
        
        # Answering should bring confidence up
        answer = manager.process_answer(question, "Yes, within last 6 hours")
        assert manager.current_state.current_confidence > 0.10
    
    def test_max_questions_approaching_limit(self, manager):
        """Test behavior as max questions limit is approached."""
        gaps = [EvidenceGap(
            evidence_type=EvidenceType.DEPLOYMENT_SCHEMA,
            missing_count=1,
            is_required=True,
            current_confidence_impact=0.0,
            potential_confidence_impact=0.15,
        )]
        manager.initialize_state(0.65, gaps)
        
        # Ask questions up to limit
        for i in range(5):
            question = manager.generate_question(gaps, manager.current_state.current_confidence)
            if question:
                manager.process_answer(question, "No")
        
        # Should now refuse to ask more
        assert manager.should_ask_question(0.65, gaps) is False


# ---------------------------------------------------------------------
# Run Tests
# ---------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

