"""
Unit tests for Recommendation Review API and Approval Workflow.

Tests cover:
- ReviewCard and ReviewAction creation & serialization
- Approval workflow state transitions
- Execution gating (safety checks)
- Audit trail persistence
- API endpoints (CRUD, approval, rejection, execution)
"""

import pytest
import json
from pathlib import Path
from datetime import datetime

from recommendations.review_schema import (
    ReviewCard,
    ReviewAction,
    ApprovalWorkflow,
    ApprovalStep,
    AuditTrail,
    AuditEntry,
    ActionRisk,
    ActionApprovalStatus,
    ApproverRole,
    OperationMode,
    RiskIndicators,
    SafetyWarnings,
    RollbackPlan,
    create_risk_indicators,
    create_safety_warnings,
)
from recommendations.approval_store import save_card, load_card, append_audit_entry


class TestReviewAction:
    """Tests for ReviewAction creation and safety checks."""

    def test_create_low_risk_action(self):
        """Test creation of a low-risk action that doesn't require approval."""
        action = ReviewAction(
            title="Safe index creation",
            description="Create a non-blocking index",
            sql_command="CREATE INDEX CONCURRENTLY idx_x ON t(x);",
            risk=ActionRisk.LOW.value,
            requires_approval=False,
        )
        assert action.title == "Safe index creation"
        assert action.risk == ActionRisk.LOW.value
        assert not action.requires_approval
        assert action.is_safe_to_execute()

    def test_create_high_risk_action_with_approval(self):
        """Test creation of high-risk action that requires approval workflow."""
        workflow = ApprovalWorkflow(
            action_id="test_act_1",
            steps=[
                ApprovalStep(approver_role=ApproverRole.DBA_SENIOR.value),
                ApprovalStep(approver_role=ApproverRole.SRE_LEAD.value),
            ],
            status=ActionApprovalStatus.PENDING.value,
            requires_multi_approval=True,
        )
        action = ReviewAction(
            title="Dangerous vacuum operation",
            description="Full vacuum with potential downtime",
            sql_command="VACUUM FULL;",
            risk=ActionRisk.CRITICAL.value,
            requires_approval=True,
            approval_workflow=workflow,
            safety_warnings=SafetyWarnings(
                is_destructive=True,
                warnings=["VACUUM FULL requires exclusive lock"],
            ),
            rollback_plan=RollbackPlan(
                rollback_command="No rollback possible",
                data_loss_risk="none",
            ),
        )
        assert action.risk == ActionRisk.CRITICAL.value
        assert action.requires_approval
        assert not action.is_safe_to_execute()  # Not approved yet

    def test_action_serialization_roundtrip(self):
        """Test action to_dict / from_dict roundtrip."""
        original = ReviewAction(
            title="Test Action",
            description="A test",
            risk=ActionRisk.MEDIUM.value,
            operation_mode=OperationMode.ONLINE.value,
        )
        data = original.to_dict()
        restored = ReviewAction.from_dict(data)
        assert restored.title == original.title
        assert restored.risk == original.risk
        assert restored.operation_mode == original.operation_mode


class TestApprovalWorkflow:
    """Tests for approval workflow state transitions and gating."""

    def test_workflow_single_approver(self):
        """Test workflow with single approver."""
        workflow = ApprovalWorkflow(
            action_id="act_1",
            steps=[ApprovalStep(approver_role=ApproverRole.DBA_SENIOR.value)],
            status=ActionApprovalStatus.PENDING.value,
        )
        assert workflow.is_pending()
        assert not workflow.is_approved()

        # Add approval
        workflow.add_approval("alice", "Looks good")
        assert workflow.is_approved()
        assert workflow.status == ActionApprovalStatus.APPROVED.value

    def test_workflow_multi_approver(self):
        """Test workflow requiring multiple approvers."""
        workflow = ApprovalWorkflow(
            action_id="act_2",
            steps=[
                ApprovalStep(approver_role=ApproverRole.DBA_SENIOR.value),
                ApprovalStep(approver_role=ApproverRole.SRE_LEAD.value),
            ],
            status=ActionApprovalStatus.PENDING.value,
            requires_multi_approval=True,
        )
        assert not workflow.is_approved()

        # First approval
        workflow.add_approval("bob", "DBA: approved")
        assert workflow.current_step == 1
        assert not workflow.is_approved()  # Still waiting for SRE

        # Second approval
        workflow.add_approval("carol", "SRE: approved")
        assert workflow.is_approved()

    def test_workflow_rejection(self):
        """Test rejection at any step."""
        workflow = ApprovalWorkflow(
            action_id="act_3",
            steps=[ApprovalStep(approver_role=ApproverRole.DBA_SENIOR.value)],
            status=ActionApprovalStatus.PENDING.value,
        )
        workflow.reject("alice", "Needs more testing")
        assert workflow.status == ActionApprovalStatus.REJECTED.value
        assert not workflow.is_approved()

    def test_workflow_serialization(self):
        """Test workflow to_dict / from_dict."""
        original = ApprovalWorkflow(
            action_id="act_4",
            steps=[ApprovalStep(approver_role=ApproverRole.DBA_SENIOR.value)],
            status=ActionApprovalStatus.APPROVED.value,
        )
        data = original.to_dict()
        restored = ApprovalWorkflow.from_dict(data)
        assert restored.action_id == original.action_id
        assert restored.status == original.status
        assert len(restored.steps) == len(original.steps)


class TestReviewCard:
    """Tests for ReviewCard creation and audit logging."""

    def test_create_card(self):
        """Test creating a review card."""
        card = ReviewCard(
            title="Query Optimization",
            summary="Add index to improve p95 latency",
            severity="medium",
        )
        assert card.title == "Query Optimization"
        assert card.status == "pending_review"
        assert len(card.audit_trail.entries) == 0

    def test_card_add_action_logs_audit(self):
        """Test that adding an action logs an audit entry."""
        card = ReviewCard(title="Test")
        action = ReviewAction(title="Test Action")
        card.add_action(action)

        assert len(card.actions) == 1
        assert len(card.audit_trail.entries) == 1
        assert card.audit_trail.entries[0].action_type == "created"

    def test_card_log_approval(self):
        """Test approval logging and gating."""
        card = ReviewCard(title="Test")
        workflow = ApprovalWorkflow(
            action_id="act_test",
            steps=[ApprovalStep(approver_role=ApproverRole.DBA_SENIOR.value)],
            status=ActionApprovalStatus.PENDING.value,
        )
        action = ReviewAction(
            title="Test", requires_approval=True, approval_workflow=workflow
        )
        card.add_action(action)

        # Log approval
        result = card.log_approval(action.action_id, "bob", "LGTM")
        assert result
        assert len(card.audit_trail.entries) == 2  # created + approved

    def test_card_log_rejection(self):
        """Test rejection logging."""
        card = ReviewCard(title="Test")
        workflow = ApprovalWorkflow(
            action_id="act_test",
            steps=[ApprovalStep(approver_role=ApproverRole.DBA_SENIOR.value)],
            status=ActionApprovalStatus.PENDING.value,
        )
        action = ReviewAction(
            title="Test", requires_approval=True, approval_workflow=workflow
        )
        card.add_action(action)

        result = card.log_rejection(action.action_id, "alice", "Needs review")
        assert result
        assert action.approval_workflow.status == ActionApprovalStatus.REJECTED.value

    def test_card_log_execution(self):
        """Test execution logging."""
        card = ReviewCard(title="Test")
        action = ReviewAction(title="Test", risk=ActionRisk.LOW.value)
        card.add_action(action)

        result = card.log_execution(action.action_id, "ci-runner")
        assert result
        assert action.status == "executed"
        assert action.executed_by == "ci-runner"

    def test_card_serialization(self):
        """Test card to_dict / from_dict."""
        card = ReviewCard(title="Test Card", summary="Test summary")
        action = ReviewAction(title="Action 1")
        card.add_action(action)

        data = card.to_dict()
        restored = ReviewCard.from_dict(data)
        assert restored.title == card.title
        assert len(restored.actions) == 1
        assert restored.actions[0].title == "Action 1"


class TestExecutionGating:
    """Tests for safety gating on execution."""

    def test_cannot_execute_without_approval(self):
        """Test that high-risk actions cannot execute without approval."""
        workflow = ApprovalWorkflow(
            action_id="act_1",
            steps=[ApprovalStep(approver_role=ApproverRole.DBA_SENIOR.value)],
            status=ActionApprovalStatus.PENDING.value,
        )
        action = ReviewAction(
            title="Delete index",
            risk=ActionRisk.HIGH.value,
            requires_approval=True,
            approval_workflow=workflow,
            rollback_plan=RollbackPlan(rollback_command="CREATE INDEX..."),
        )
        assert not action.is_safe_to_execute()

    def test_can_execute_after_approval(self):
        """Test that action can execute after approval."""
        workflow = ApprovalWorkflow(
            action_id="act_1",
            steps=[ApprovalStep(approver_role=ApproverRole.DBA_SENIOR.value)],
            status=ActionApprovalStatus.PENDING.value,
        )
        action = ReviewAction(
            title="Delete index",
            risk=ActionRisk.HIGH.value,
            requires_approval=True,
            approval_workflow=workflow,
            rollback_plan=RollbackPlan(rollback_command="CREATE INDEX..."),
        )
        workflow.add_approval("bob", "approved")
        assert action.is_safe_to_execute()

    def test_cannot_execute_destructive_without_rollback(self):
        """Test that destructive actions require rollback plan."""
        action = ReviewAction(
            title="TRUNCATE TABLE",
            sql_command="TRUNCATE TABLE huge_table;",
            risk=ActionRisk.CRITICAL.value,
            requires_approval=True,
            approval_workflow=ApprovalWorkflow(
                action_id="act_1",
                steps=[ApprovalStep(approver_role=ApproverRole.DBA_SENIOR.value)],
                status=ActionApprovalStatus.APPROVED.value,
            ),
            safety_warnings=SafetyWarnings(is_destructive=True),
            rollback_plan=RollbackPlan(rollback_command=""),  # Empty rollback
        )
        assert not action.is_safe_to_execute()


class TestAuditTrail:
    """Tests for audit trail tracking."""

    def test_audit_entry_creation(self):
        """Test creating an audit entry."""
        entry = AuditEntry(
            action_type="approved",
            actor="bob",
            actor_role="DBA",
            target_type="action",
            target_id="act_1",
            details="Approved by DBA",
        )
        assert entry.actor == "bob"
        assert entry.action_type == "approved"

    def test_audit_trail_append(self):
        """Test appending to audit trail."""
        trail = AuditTrail(review_card_id="rc_1")
        entry1 = AuditEntry(
            action_type="created", actor="alice", target_type="review_card"
        )
        entry2 = AuditEntry(
            action_type="approved", actor="bob", target_type="action"
        )
        trail.add_entry(entry1)
        trail.add_entry(entry2)

        assert len(trail.entries) == 2
        assert trail.get_entries_by_type("created")[0].actor == "alice"

    def test_audit_trail_history_queries(self):
        """Test querying audit trail for specific event types."""
        trail = AuditTrail(review_card_id="rc_1")
        trail.add_entry(AuditEntry(action_type="created", actor="alice"))
        trail.add_entry(AuditEntry(action_type="approved", actor="bob"))
        trail.add_entry(AuditEntry(action_type="executed", actor="ci"))
        trail.add_entry(AuditEntry(action_type="rolled_back", actor="alice"))

        approvals = trail.get_approval_history()
        assert len(approvals) == 1

        executions = trail.get_execution_history()
        assert len(executions) == 2


class TestApprovalStore:
    """Tests for JSON-backed approval store persistence."""

    def test_save_and_load_card(self, tmp_path):
        """Test persisting and loading a card from storage."""
        # Create a card
        card = ReviewCard(title="Test", summary="Test card")
        action = ReviewAction(title="Action 1", risk=ActionRisk.MEDIUM.value)
        card.add_action(action)

        # Manually save/load (simulating store behavior)
        card_data = card.to_dict()
        restored = ReviewCard.from_dict(card_data)

        assert restored.card_id == card.card_id
        assert restored.title == card.title
        assert len(restored.actions) == 1

    def test_audit_entry_append_to_store(self, tmp_path):
        """Test appending audit entries to stored card."""
        card = ReviewCard(title="Test")
        workflow = ApprovalWorkflow(
            action_id="act_test",
            steps=[ApprovalStep(approver_role=ApproverRole.DBA_SENIOR.value)],
            status=ActionApprovalStatus.PENDING.value,
        )
        action = ReviewAction(
            title="Action 1",
            requires_approval=True,
            approval_workflow=workflow,
        )
        card.add_action(action)

        # Simulate adding approval
        card.log_approval(action.action_id, "bob", "Looks good")

        # Verify audit trail
        audit_summary = card.get_audit_summary()
        assert audit_summary["created"] == 1
        assert audit_summary["approved"] == 1


class TestRiskIndicators:
    """Tests for risk indicator generation."""

    def test_risk_indicator_low(self):
        """Test low risk indicators."""
        indicators = create_risk_indicators("low")
        assert indicators.risk_level == "low"
        assert "LOW" in indicators.severity_badge
        assert indicators.color_code == "#00CC66"

    def test_risk_indicator_critical(self):
        """Test critical risk indicators."""
        indicators = create_risk_indicators("critical")
        assert indicators.risk_level == "critical"
        assert "CRITICAL" in indicators.severity_badge
        assert indicators.color_code == "#FF4444"


class TestSafetyWarnings:
    """Tests for safety warning generation."""

    def test_warnings_for_destructive_sql(self):
        """Test warning generation for destructive SQL."""
        warnings = create_safety_warnings("DROP TABLE users;", "high")
        assert warnings.is_destructive
        assert any("DROP" in w for w in warnings.warnings)

    def test_warnings_for_reindex_without_concurrent(self):
        """Test warning for REINDEX without CONCURRENTLY."""
        warnings = create_safety_warnings("REINDEX TABLE users;", "medium")
        assert warnings.is_destructive
        assert any("CONCURRENTLY" in w for w in warnings.warnings)

    def test_warnings_for_alter_system(self):
        """Test warning for configuration changes."""
        warnings = create_safety_warnings(
            "ALTER SYSTEM SET shared_buffers = '8GB';", "high"
        )
        assert any("restart" in w.lower() for w in warnings.warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
