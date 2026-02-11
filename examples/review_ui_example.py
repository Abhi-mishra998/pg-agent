#!/usr/bin/env python3
"""
review_ui_example.py

Example demonstrating the Recommendation Review UI for PostgreSQL DBA Safety

This example creates sample review cards with:
- Various risk levels (low, medium, high, critical)
- Different approval workflows
- Complete safety information
- Audit trail logging
- Multiple output formats (terminal, markdown, HTML, JSON)
"""

import json
from datetime import datetime

from recommendations.review_schema import (
    ReviewCard,
    ReviewAction,
    ApprovalWorkflow,
    ApprovalStep,
    AuditTrail,
    AuditEntry,
    AuditActionType,
    ActionRisk,
    ActionApprovalStatus,
    OperationMode,
    ImpactScope,
    RiskIndicators,
    SafetyWarnings,
    RollbackPlan,
    create_risk_indicators,
    create_safety_warnings,
    create_rollback_plan,
    build_approval_workflow,
    ReviewSession,
)

from recommendations.review_renderer import (
    render_to_terminal,
    render_to_markdown,
    render_to_html,
    render_to_json,
    render_session_to_terminal,
    render_session_to_markdown,
    render_session_to_html,
)


def create_sample_review_card() -> ReviewCard:
    """Create a comprehensive sample review card with multiple actions."""
    
    # Create actions with different risk levels
    actions = []
    
    # Action 1: LOW risk - ANALYZE (safe, no approval needed)
    analyze_action = ReviewAction(
        action_id="act_analyze_001",
        action_type="ANALYZE",
        title="Run ANALYZE on orders_table",
        description="Update table statistics to improve query planning accuracy. This is a read-only operation that only collects statistics.",
        sql_command="ANALYZE orders_table;",
        priority="high",
        estimated_duration="~10 seconds",
        risk=ActionRisk.LOW.value,
        risk_indicators=create_risk_indicators(ActionRisk.LOW.value),
        safety_warnings=create_safety_warnings("ANALYZE orders_table;", ActionRisk.LOW.value),
        operation_mode=OperationMode.ONLINE.value,
        requires_approval=False,
        rollback_plan=RollbackPlan(
            rollback_command="Not applicable - ANALYZE only collects statistics",
            recovery_time_estimate="Instant",
            data_loss_risk="none",
        ),
        impact_scope=ImpactScope.TABLE.value,
        affected_objects=["orders_table"],
    )
    actions.append(analyze_action)
    
    # Action 2: MEDIUM risk - CREATE INDEX (needs caution)
    create_index_action = ReviewAction(
        action_id="act_create_index_001",
        action_type="CREATE_INDEX",
        title="Create index on customer_id and created_at",
        description="Create a composite index to eliminate sequential scans on the orders_table query. Uses CONCURRENTLY to avoid locks.",
        sql_command="""-- Create index concurrently to avoid production locks
CREATE INDEX CONCURRENTLY 
  idx_orders_customer_created 
ON orders_table (customer_id, created_at);

-- Verify index was created
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename = 'orders_table';""",
        priority="critical",
        estimated_duration="~5 minutes (for 10M rows)",
        risk=ActionRisk.MEDIUM.value,
        risk_indicators=create_risk_indicators(ActionRisk.MEDIUM.value),
        safety_warnings=create_safety_warnings(
            "CREATE INDEX CONCURRENTLY ON orders_table (customer_id, created_at);",
            ActionRisk.MEDIUM.value
        ),
        operation_mode=OperationMode.CONCURRENTLY.value,
        requires_approval=True,
        approval_workflow=build_approval_workflow(
            "act_create_index_001",
            ActionRisk.MEDIUM.value,
            requires_approval=True
        ),
        rollback_plan=RollbackPlan(
            rollback_command="DROP INDEX CONCURRENTLY idx_orders_customer_created;",
            recovery_time_estimate="~30 seconds",
            data_loss_risk="none",
        ),
        impact_scope=ImpactScope.TABLE.value,
        affected_objects=["orders_table"],
        kb_entry_id="kb_missing_index_001",
        references=["PostgreSQL CREATE INDEX documentation"],
    )
    actions.append(create_index_action)
    
    # Action 3: HIGH risk - VACUUM FULL (requires approval)
    vacuum_full_action = ReviewAction(
        action_id="act_vacuum_full_001",
        action_type="VACUUM_FULL",
        title="Reclaim disk space with VACUUM FULL",
        description="VACUUM FULL will rewrite the entire table to reclaim space. This requires an exclusive lock and can take significant time.",
        sql_command="VACUUM FULL orders_table;",
        priority="low",
        estimated_duration="~30 minutes (for 10GB table)",
        risk=ActionRisk.HIGH.value,
        risk_indicators=create_risk_indicators(ActionRisk.HIGH.value),
        safety_warnings=SafetyWarnings(
            is_destructive=True,
            warnings=[
                "VACUUM FULL requires exclusive table lock",
                "Table will be inaccessible during operation",
                "Can take significant time on large tables",
            ],
            preconditions=[
                "No long-running transactions active",
                "Maintenance window scheduled",
                "Backup completed",
            ],
        ),
        operation_mode=OperationMode.OFFLINE.value,
        requires_maintenance_window=True,
        requires_approval=True,
        approval_workflow=build_approval_workflow(
            "act_vacuum_full_001",
            ActionRisk.HIGH.value,
            requires_approval=True
        ),
        rollback_plan=RollbackPlan(
            rollback_command="Not applicable - VACUUM FULL is recovery operation",
            recovery_time_estimate="N/A",
            data_loss_risk="none",
        ),
        impact_scope=ImpactScope.TABLE.value,
        affected_objects=["orders_table"],
    )
    actions.append(vacuum_full_action)
    
    # Action 4: CRITICAL risk - DROP INDEX (dangerous)
    drop_index_action = ReviewAction(
        action_id="act_drop_index_001",
        action_type="DROP_INDEX",
        title="Remove unused index idx_old_customers",
        description="Drop an index that has not been used in the past 30 days. This will improve write performance but may impact read queries.",
        sql_command="DROP INDEX CONCURRENTLY idx_old_customers;",
        priority="medium",
        estimated_duration="~10 seconds",
        risk=ActionRisk.CRITICAL.value,
        risk_indicators=create_risk_indicators(ActionRisk.CRITICAL.value),
        safety_warnings=SafetyWarnings(
            is_destructive=True,
            warnings=[
                "DROP INDEX will remove index structure permanently",
                "Queries using this index will fail or use sequential scan",
                "Verify index is truly unused before dropping",
            ],
            preconditions=[
                "Confirm index not used in pg_stat_user_indexes",
                "Check query patterns don't rely on this index",
                "Have recreation SQL ready",
            ],
            conflicts=["Any queries filtering on indexed columns"],
        ),
        operation_mode=OperationMode.CONCURRENTLY.value,
        requires_approval=True,
        approval_workflow=build_approval_workflow(
            "act_drop_index_001",
            ActionRisk.CRITICAL.value,
            requires_approval=True
        ),
        rollback_plan=RollbackPlan(
            rollback_command="CREATE INDEX CONCURRENTLY idx_old_customers ON {table_name} ({columns});",
            recovery_time_estimate="~5 minutes",
            data_loss_risk="none",
        ),
        impact_scope=ImpactScope.TABLE.value,
        affected_objects=["orders_table", "idx_old_customers"],
    )
    actions.append(drop_index_action)
    
    # Action 5: Configuration change (HIGH risk)
    config_change_action = ReviewAction(
        action_id="act_config_change_001",
        action_type="CONFIG_CHANGE",
        title="Increase shared_buffers to 24GB",
        description="Increase shared_buffers from 16GB to 24GB to improve cache performance for large working sets.",
        sql_command="""-- This requires PostgreSQL restart
ALTER SYSTEM SET shared_buffers = '24GB';

-- Then restart PostgreSQL:
-- sudo systemctl restart postgresql""",
        config_change={
            "parameter": "shared_buffers",
            "current_value": "16GB",
            "new_value": "24GB",
            "requires_restart": True,
        },
        priority="low",
        estimated_duration="~5 minutes + restart time",
        risk=ActionRisk.HIGH.value,
        risk_indicators=create_risk_indicators(ActionRisk.HIGH.value),
        safety_warnings=SafetyWarnings(
            warnings=[
                "Configuration change requires PostgreSQL restart",
                "Too high shared_buffers can cause performance issues",
                "Recommend testing in staging first",
            ],
            preconditions=[
                "Maintenance window scheduled",
                "DBA team notified",
                "Rollback value documented",
            ],
        ),
        operation_mode=OperationMode.OFFLINE.value,
        requires_maintenance_window=True,
        requires_approval=True,
        approval_workflow=build_approval_workflow(
            "act_config_change_001",
            ActionRisk.HIGH.value,
            requires_approval=True
        ),
        rollback_plan=RollbackPlan(
            rollback_command="ALTER SYSTEM SET shared_buffers = '16GB'; -- Requires restart",
            recovery_time_estimate="~2 minutes + restart time",
            data_loss_risk="none",
        ),
        impact_scope=ImpactScope.DATABASE.value,
        affected_objects=["production_orders"],
    )
    actions.append(config_change_action)
    
    # Create audit trail
    audit_trail = AuditTrail()
    audit_trail.add_entry(AuditEntry(
        action_type=AuditActionType.CREATED.value,
        actor="system",
        target_type="review_card",
        details="Review card created from incident INC-20260125-8F2A1B"
    ))
    audit_trail.add_entry(AuditEntry(
        action_type=AuditActionType.VIEWED.value,
        actor="dba_john",
        target_type="review_card",
        details="Initial review by DBA John"
    ))
    
    # Create review card
    card = ReviewCard(
        card_id="rc_20260125_001",
        incident_id="INC-20260125-8F2A1B",
        title="QUERY_PERFORMANCE: Sequential Scan on orders_table",
        summary="Multiple recommendations to resolve sequential scan issue causing 15s+ query times. Primary recommendation is creating a composite index on (customer_id, created_at).",
        category="query_performance",
        severity="high",
        actions=actions,
        confidence_score=0.92,
        evidence_count=7,
        root_cause="MISSING_INDEX: Query on orders_table lacks index on (customer_id, created_at) columns causing sequential scan",
        audit_trail=audit_trail,
        status="pending_review",
    )
    
    return card


def create_review_session() -> ReviewSession:
    """Create a review session with multiple cards."""
    
    session = ReviewSession(
        session_id="session_20260125_001",
        created_by="dba_john",
    )
    
    # Add primary card
    card = create_sample_review_card()
    session.add_card(card)
    
    return session


def demonstrate_terminal_output():
    """Demonstrate terminal rendering."""
    print("\n" + "=" * 80)
    print(" TERMINAL OUTPUT DEMONSTRATION")
    print("=" * 80 + "\n")
    
    card = create_sample_review_card()
    output = render_to_terminal(card, use_colors=True)
    print(output)


def demonstrate_markdown_output():
    """Demonstrate markdown rendering."""
    print("\n" + "=" * 80)
    print(" MARKDOWN OUTPUT DEMONSTRATION")
    print("=" * 80 + "\n")
    
    card = create_sample_review_card()
    output = render_to_markdown(card)
    print(output)


def demonstrate_html_output():
    """Demonstrate HTML rendering."""
    print("\n" + "=" * 80)
    print(" HTML OUTPUT DEMONSTRATION")
    print("=" * 80 + "\n")
    
    card = create_sample_review_card()
    output = render_to_html(card)
    
    # Save to file
    output_path = "/Users/abhishekmishra/pg-agent/data/output/review_card.html"
    with open(output_path, 'w') as f:
        f.write(output)
    print(f"HTML output saved to: {output_path}")
    print(f"Output length: {len(output)} characters")


def demonstrate_json_output():
    """Demonstrate JSON rendering."""
    print("\n" + "=" * 80)
    print(" JSON OUTPUT DEMONSTRATION")
    print("=" * 80 + "\n")
    
    card = create_sample_review_card()
    output = render_to_json(card)
    
    # Save to file
    output_path = "/Users/abhishekmishra/pg-agent/data/output/review_card.json"
    with open(output_path, 'w') as f:
        f.write(output)
    print(f"JSON output saved to: {output_path}")


def demonstrate_approval_workflow():
    """Demonstrate approval workflow functionality."""
    print("\n" + "=" * 80)
    print(" APPROVAL WORKFLOW DEMONSTRATION")
    print("=" * 80 + "\n")
    
    card = create_sample_review_card()
    
    # Find action requiring approval
    for action in card.actions:
        if action.requires_approval and action.approval_workflow:
            print(f"Action: {action.title}")
            print(f"  Risk: {action.risk.upper()}")
            print(f"  Approval Workflow Status: {action.approval_workflow.status}")
            print(f"  Steps: {len(action.approval_workflow.steps)}")
            print()
            
            # Simulate approvals
            print("Simulating approval workflow...")
            workflow = action.approval_workflow
            
            for step in workflow.steps:
                print(f"  Step {step.step_number}: {step.approver_role} - {step.status}")
            
            # Add approval
            workflow.add_approval("dba_lead_mary", "Approved - verified index is safe")
            print(f"\nAfter approval:")
            print(f"  Status: {workflow.status}")
            print(f"  Current Step: {workflow.current_step + 1}/{len(workflow.steps)}")
            
            if workflow.is_approved():
                print(f"  ✅ Action is now approved for execution!")
            
            break


def demonstrate_audit_trail():
    """Demonstrate audit trail functionality."""
    print("\n" + "=" * 80)
    print(" AUDIT TRAIL DEMONSTRATION")
    print("=" * 80 + "\n")
    
    card = create_sample_review_card()
    
    print(f"Review Card: {card.card_id}")
    print(f"Total Audit Entries: {len(card.audit_trail.entries)}")
    print()
    
    print("Audit Trail Entries:")
    for entry in card.audit_trail.entries:
        print(f"  [{entry.timestamp}] {entry.action_type}: {entry.details}")
    
    print()
    
    # Show audit summary
    summary = card.get_audit_summary()
    print("Audit Summary:")
    for action_type, count in summary.items():
        if count > 0:
            print(f"  {action_type}: {count}")


def demonstrate_safety_features():
    """Demonstrate safety features."""
    print("\n" + "=" * 80)
    print(" SAFETY FEATURES DEMONSTRATION")
    print("=" * 80 + "\n")
    
    card = create_sample_review_card()
    
    print("Risk Analysis:")
    print("-" * 40)
    
    for action in card.actions:
        print(f"\n{action.title}")
        print(f"  Risk Level: {action.risk.upper()}")
        print(f"  Risk Summary: {action.risk_indicators.risk_summary}")
        print(f"  Operation Mode: {action.operation_mode}")
        print(f"  Requires Approval: {action.requires_approval}")
        print(f"  Is Safe to Execute: {action.is_safe_to_execute()}")
        
        if action.safety_warnings.is_destructive:
            print(f"  ⚠️  DANGEROUS ACTION - Data Risk: {action.rollback_plan.data_loss_risk}")
        
        if action.safety_warnings.warnings:
            print(f"  Warnings: {len(action.safety_warnings.warnings)}")
            for w in action.safety_warnings.warnings[:2]:
                print(f"    - {w}")
    
    print("\n" + "-" * 40)
    print("\nSummary:")
    high_risk = card.get_high_risk_actions()
    approval_required = card.get_actions_requiring_approval()
    offline_actions = card.get_offline_actions()
    
    print(f"  Total Actions: {len(card.actions)}")
    print(f"  High/Critical Risk: {len(high_risk)}")
    print(f"  Requiring Approval: {len(approval_required)}")
    print(f"  Offline/Maintenance: {len(offline_actions)}")


def demonstrate_session_output():
    """Demonstrate full session rendering."""
    print("\n" + "=" * 80)
    print(" FULL SESSION OUTPUT DEMONSTRATION")
    print("=" * 80 + "\n")
    
    session = create_review_session()
    
    # Save session as JSON
    session_path = "/Users/abhishekmishra/pg-agent/data/output/review_session.json"
    with open(session_path, 'w') as f:
        f.write(render_session_to_json(session))
    print(f"Session JSON saved to: {session_path}")
    
    # Save session as HTML
    session_html_path = "/Users/abhishekmishra/pg-agent/data/output/review_session.html"
    with open(session_html_path, 'w') as f:
        f.write(render_session_to_html(session))
    print(f"Session HTML saved to: {session_html_path}")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 80)
    print("#  RECOMMENDATION REVIEW UI - COMPLETE DEMONSTRATION")
    print("#" * 80)
    
    # Run all demonstrations
    demonstrate_terminal_output()
    demonstrate_markdown_output()
    demonstrate_html_output()
    demonstrate_json_output()
    demonstrate_approval_workflow()
    demonstrate_audit_trail()
    demonstrate_safety_features()
    demonstrate_session_output()
    
    print("\n" + "#" * 80)
    print("#  DEMONSTRATION COMPLETE")
    print("#" * 80)
    print("\nOutput files saved to: /Users/abhishekmishra/pg-agent/data/output/")
    print("  - review_card.html")
    print("  - review_card.json")
    print("  - review_session.json")
    print("  - review_session.html")


if __name__ == "__main__":
    main()
