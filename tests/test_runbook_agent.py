#!/usr/bin/env python3
"""
Tests for RunbookAgent with RAG functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.runbook_agent import RunbookAgent, RunbookOutput, KBContextEntry


class TestRunbookOutput:
    """Test cases for RunbookOutput dataclass."""
    
    def test_default_values(self):
        """Test that default values are empty strings."""
        output = RunbookOutput()
        assert output.diagnosis == ""
        assert output.evidence == ""
        assert output.root_cause == ""
        assert output.immediate_actions == ""
        assert output.preventive_actions == ""
        assert output.cli_commands == ""
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        output = RunbookOutput(
            diagnosis="Test diagnosis",
            evidence="Test evidence",
            root_cause="Test root cause",
            immediate_actions="1. Action 1",
            preventive_actions="1. Prevention 1",
            cli_commands="SELECT 1;",
        )
        
        result = output.to_dict()
        
        assert result["diagnosis"] == "Test diagnosis"
        assert result["evidence"] == "Test evidence"
        assert result["root_cause"] == "Test root cause"
        assert result["immediate_actions"] == "1. Action 1"
        assert result["preventive_actions"] == "1. Prevention 1"
        assert result["cli_commands"] == "SELECT 1;"
    
    def test_to_markdown(self):
        """Test markdown formatting."""
        output = RunbookOutput(
            diagnosis="Test diagnosis",
            evidence="Test evidence",
            root_cause="Test root cause",
            immediate_actions="1. Action 1",
            preventive_actions="1. Prevention 1",
            cli_commands="SELECT 1;",
        )
        
        markdown = output.to_markdown()
        
        assert "## Diagnosis" in markdown
        assert "Test diagnosis" in markdown
        assert "## Evidence" in markdown
        assert "## Root Cause" in markdown
        assert "## Immediate Actions" in markdown
        assert "## Preventive Actions" in markdown
        assert "## CLI Commands" in markdown
    
    def test_is_empty(self):
        """Test empty check."""
        empty_output = RunbookOutput()
        assert empty_output.is_empty() is True
        
        non_empty = RunbookOutput(diagnosis="Some diagnosis")
        assert non_empty.is_empty() is False


class TestRunbookAgent:
    """Test cases for RunbookAgent."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        mock = Mock()
        mock.is_available.return_value = True
        mock.generate.return_value = Mock(
            text="""Diagnosis:
Test diagnosis for slow query.

Evidence:
- Sequential scan detected on orders table
- Table has 10M rows
- Query latency 15s

Root Cause:
Missing index on customer_id column.

Immediate Actions:
1. Create index on customer_id
2. Run EXPLAIN ANALYZE

Preventive Actions:
1. Add index review to deployment
2. Monitor sequential scans

CLI Commands:
```sql
CREATE INDEX idx ON orders(customer_id);
```"""
        )
        return mock
    
    @pytest.fixture
    def runbook_agent(self, mock_llm):
        """Create RunbookAgent with mocked dependencies."""
        with patch('agent.runbook_agent.LlamaClient', return_value=mock_llm):
            agent = RunbookAgent(
                rules_path="model/rules.md",
                examples_path="model/examples.md",
                kb_path="data/kb_unified.json",
            )
            return agent
    
    def test_initialization(self, runbook_agent):
        """Test agent initialization."""
        assert runbook_agent.llm is not None
        assert runbook_agent.rules != ""
        assert runbook_agent.examples != ""
    
    def test_is_available(self, runbook_agent):
        """Test LLM availability check."""
        assert runbook_agent.is_available() is True
    
    def test_parse_response(self, runbook_agent):
        """Test parsing LLM response into RunbookOutput."""
        response = """Diagnosis:
Test diagnosis.

Evidence:
Test evidence.

Root Cause:
Test root cause.

Immediate Actions:
1. Action 1
2. Action 2

Preventive Actions:
1. Prevention 1

CLI Commands:
```sql
SELECT 1;
```"""
        
        output = runbook_agent._parse_response(response)
        
        assert isinstance(output, RunbookOutput)
        assert output.diagnosis == "Test diagnosis."
        assert output.evidence == "Test evidence."
        assert output.root_cause == "Test root cause."
        assert "Action 1" in output.immediate_actions
        assert "Action 2" in output.immediate_actions
        assert "SELECT 1" in output.cli_commands
    
    def test_parse_response_with_numbered_lists(self, runbook_agent):
        """Test parsing response with numbered lists."""
        response = """Diagnosis:
Diagnosis without numbering.

Evidence:
Evidence without numbering.

Root Cause:
Root cause without numbering.

Immediate Actions:
First action
Second action
Third action

Preventive Actions:
Prevention 1
Prevention 2

CLI Commands:
```sql
SELECT 1;
```"""
        
        output = runbook_agent._parse_response(response)
        
        # Check that numbered lists were added
        assert output.immediate_actions.startswith("1.")
        assert "First action" in output.immediate_actions
    
    def test_fallback_response(self, runbook_agent):
        """Test fallback response when LLM fails."""
        output = runbook_agent._fallback_response()
        
        assert isinstance(output, RunbookOutput)
        assert "LLM service unavailable" in output.diagnosis
        assert "Ollama" in output.cli_commands
    
    def test_generate_from_query(self, runbook_agent):
        """Test generating runbook from natural language query."""
        result = runbook_agent.generate_from_query(
            query="Sequential scan on large table",
        )
        
        assert isinstance(result, RunbookOutput)
        # The mock response should be parsed
        assert result.diagnosis != ""
    
    def test_generate_from_signals(self, runbook_agent):
        """Test generating runbook from signal list."""
        signals = [
            {
                "name": "sequential_scan_detected",
                "type": "sql",
                "severity": "high",
                "data": {"table": "orders", "seq_scan_ratio": 0.99},
            },
            {
                "name": "high_query_latency",
                "type": "query_metrics",
                "severity": "high",
                "data": {"latency_ms": 15000},
            },
        ]
        
        result = runbook_agent.generate_from_signals(signals=signals)
        
        assert isinstance(result, RunbookOutput)
        assert result.diagnosis != ""
    
    def test_build_prompt(self, runbook_agent):
        """Test prompt building."""
        context = {
            "signals": [{"name": "test_signal"}],
            "metrics": {"confidence": 0.9},
        }
        kb_context = [
            KBContextEntry(
                kb_id="test_001",
                title="Test Issue",
                description="Test description",
                severity="high",
                category="query_performance",
                recommendations=["Test recommendation"],
            )
        ]
        
        prompt = runbook_agent._build_prompt(
            context=context,
            kb_context=kb_context,
            user_question="Test question?",
        )
        
        assert "CONTEXT FROM ANALYSIS" in prompt
        assert "Test question?" in prompt
        assert "KB Entry: test_001" in prompt
        assert "Diagnosis:" in prompt
        assert "Evidence:" in prompt
        assert "Root Cause:" in prompt


class TestKBContextEntry:
    """Test cases for KBContextEntry."""
    
    def test_create_with_all_fields(self):
        """Test creating entry with all fields."""
        entry = KBContextEntry(
            kb_id="test_001",
            title="Test Issue",
            description="Test description",
            severity="high",
            category="query_performance",
            recommendations=["Rec 1", "Rec 2"],
            evidence=["Evidence 1"],
        )
        
        assert entry.kb_id == "test_001"
        assert entry.title == "Test Issue"
        assert entry.severity == "high"
        assert len(entry.recommendations) == 2
        assert len(entry.evidence) == 1
    
    def test_create_with_defaults(self):
        """Test creating entry with default fields."""
        entry = KBContextEntry(
            kb_id="test_002",
            title="Test Issue",
            description="Test description",
            severity="medium",
            category="maintenance",
        )
        
        assert entry.recommendations == []
        assert entry.evidence == []


class TestResponseParsing:
    """Test response parsing edge cases."""
    
    @pytest.fixture
    def runbook_agent(self):
        """Create RunbookAgent without mocking LLM for parsing tests."""
        with patch('agent.runbook_agent.LlamaClient') as mock_llm:
            mock_llm.return_value.is_available.return_value = True
            agent = RunbookAgent()
            return agent
    
    def test_parse_partial_response(self, runbook_agent):
        """Test parsing response with only some sections."""
        response = """Diagnosis:
Partial diagnosis.

Evidence:
Some evidence.
"""
        
        output = runbook_agent._parse_response(response)
        
        assert output.diagnosis == "Partial diagnosis."
        assert output.evidence == "Some evidence."
        # Other fields should be empty
        assert output.root_cause == ""
    
    def test_parse_empty_response(self, runbook_agent):
        """Test parsing empty response."""
        output = runbook_agent._parse_response("")
        
        assert output.is_empty() is True
    
    def test_parse_multiline_sections(self, runbook_agent):
        """Test parsing response with multiline sections."""
        response = """Diagnosis:
This is a
multiline diagnosis
with multiple lines.

Evidence:
- Evidence item 1
- Evidence item 2
- Evidence item 3

Root Cause:
This is the root cause
spanning multiple lines.
"""
        
        output = runbook_agent._parse_response(response)
        
        assert "multiline diagnosis" in output.diagnosis
        assert "Evidence item 1" in output.evidence
        assert "root cause" in output.root_cause.lower()


class TestIntegration:
    """Integration tests."""
    
    def test_agent_with_real_files(self):
        """Test agent loads files correctly."""
        with patch('agent.runbook_agent.LlamaClient') as mock_llm:
            mock_llm.return_value.is_available.return_value = True
            agent = RunbookAgent(
                rules_path="model/rules.md",
                examples_path="model/examples.md",
            )
            
            # Check that files were loaded
            assert "ROLE" in agent.rules
            assert "TRAINING EXAMPLES" in agent.examples
    
    def test_agent_without_kb(self):
        """Test agent handles missing KB gracefully."""
        with patch('agent.runbook_agent.LlamaClient') as mock_llm:
            mock_llm.return_value.is_available.return_value = True
            
            # Create with non-existent KB
            agent = RunbookAgent(
                kb_path="nonexistent/kb.json",
            )
            
            # KB should be None or empty
            # The agent should still work without KB
            assert agent.kb is None or hasattr(agent.kb, 'entry_count')


# =====================================================================
# EXAMPLE USAGE
# =====================================================================

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

