#!/usr/bin/env python3
"""
review_schema.py

Recommendation Review UI Schema for PostgreSQL DBA Safety

Provides data models for:
- ReviewAction: Actions with full safety information
- ReviewCard: Complete recommendation card design
- ApprovalWorkflow: Multi-level approval state machine
- AuditEntry: Complete audit trail tracking

Safety Features:
- Risk level indicators (low/medium/high/critical)
- Online vs offline operation indicators
- Approval requirements with role mapping
- Rollback plans with recovery commands
- SQL preview (read-only with copy protection)
- Destructive action warnings

Design Principles:
- Prevent accidental destructive actions
- Encourage peer review
- Log all approvals and rejections
- Visual cues for dangerous operations
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from recommendations.approval_store import save_card, append_audit_entry


# =====================================================================
# ENUMS
# =====================================================================

class ActionRisk(Enum):
    """Risk level for recommended actions."""
    LOW = "low"              # Safe, reversible, no locks
    MEDIUM = "medium"        # Some risk, requires care
    HIGH = "high"            # Significant risk, needs approval
    CRITICAL = "critical"    # Very risky, must have approval


class ActionApprovalStatus(Enum):
    """Approval status for actions requiring authorization."""
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class ApproverRole(Enum):
    """Roles that can approve actions."""
    DBA_SENIOR = "DBA_SENIOR"        # Senior DBA
    DBA_LEAD = "DBA_LEAD"            # DBA Team Lead
    SRE_LEAD = "SRE_LEAD"            # SRE Team Lead
    MANAGER = "MANAGER"              # Engineering Manager
    DIRECTOR = "DIRECTOR"            # Director level
    SELF_APPROVED = "self_approved"  # Auto-approved low-risk actions


class ImpactScope(Enum):
    """Scope of impact for an action."""
    QUERY = "query"                  # Single query level
    TABLE = "table"                  # Table level
    SCHEMA = "schema"                # Schema level
    DATABASE = "database"            # Database level
    CLUSTER = "cluster"              # Entire cluster


class OperationMode(Enum):
    """Whether action can run online or requires downtime."""
    ONLINE = "online"                # Safe to run while database is active
    OFFLINE = "offline"              # Requires maintenance window
    CONCURRENTLY = "concurrently"    # Safe with CONCURRENTLY option


class AuditActionType(Enum):
    """Types of audit log entries."""
    CREATED = "created"              # Review card created
    VIEWED = "viewed"                # Card was viewed
    APPROVED = "approved"            # Action approved
    REJECTED = "rejected"            # Action rejected
    EXECUTED = "executed"            # Action was executed
    ROLLED_BACK = "rolled_back"      # Action was rolled back
    ESCALATED = "escalated"          # Escalated to higher approver
    COMMENTED = "commented"          # Comment added
    MODIFIED = "modified"            # Action was modified


# =====================================================================
# SAFETY & RISK MODELS
# =====================================================================

@dataclass
class RiskIndicators:
    """Visual and textual risk indicators for an action."""
    risk_level: str = ActionRisk.LOW.value
    severity_badge: str = "🟢 LOW"     # Visual badge for UI
    color_code: str = "#00CC66"        # Hex color for UI
    risk_summary: str = ""             # Short risk description
    risk_details: List[str] = field(default_factory=list)  # Detailed risks
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "RiskIndicators":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SafetyWarnings:
    """Safety warnings and caveats for an action."""
    is_destructive: bool = False
    warnings: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)  # Must be true before execution
    conflicts: List[str] = field(default_factory=list)      # Actions that conflict
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v or k in ['is_destructive']}
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SafetyWarnings":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RollbackPlan:
    """Complete rollback plan for an action."""
    rollback_command: str = ""
    rollback_sql: Optional[str] = None
    rollback_steps: List[str] = field(default_factory=list)
    recovery_time_estimate: str = ""  # e.g., "~30 seconds"
    data_loss_risk: str = "none"      # none/minor/major
    verification_command: str = ""    # Command to verify rollback
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v or k in ['data_loss_risk']}
    
    @classmethod
    def from_dict(cls, data: Dict) -> "RollbackPlan":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =====================================================================
# APPROVAL WORKFLOW
# =====================================================================

@dataclass
class ApprovalStep:
    """A single approval step in the workflow."""
    step_id: str = field(default_factory=lambda: f"step_{uuid.uuid4().hex[:8]}")
    step_number: int = 1
    approver_role: str = ApproverRole.DBA_SENIOR.value
    approver_name: Optional[str] = None
    status: str = ActionApprovalStatus.PENDING.value
    requested_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    responded_at: Optional[str] = None
    comments: str = ""
    escalation_reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ApprovalStep":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ApprovalWorkflow:
    """Multi-level approval workflow for actions."""
    workflow_id: str = field(default_factory=lambda: f"wf_{uuid.uuid4().hex[:8]}")
    action_id: str = ""
    steps: List[ApprovalStep] = field(default_factory=list)
    current_step: int = 0
    status: str = ActionApprovalStatus.NOT_REQUIRED.value
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    requires_multi_approval: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ApprovalWorkflow":
        steps = [ApprovalStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            workflow_id=data.get("workflow_id", ""),
            action_id=data.get("action_id", ""),
            steps=steps,
            current_step=data.get("current_step", 0),
            status=data.get("status", ActionApprovalStatus.NOT_REQUIRED.value),
            created_at=data.get("created_at", ""),
            completed_at=data.get("completed_at"),
            requires_multi_approval=data.get("requires_multi_approval", False),
        )
    
    def is_approved(self) -> bool:
        """Check if workflow is fully approved."""
        return self.status == ActionApprovalStatus.APPROVED.value
    
    def is_pending(self) -> bool:
        """Check if workflow is pending approval."""
        return self.status == ActionApprovalStatus.PENDING.value
    
    def get_current_approvers(self) -> List[str]:
        """Get list of approvers for current step."""
        if self.current_step < len(self.steps):
            return [self.steps[self.current_step].approver_role]
        return []
    
    def add_approval(self, approver_name: str, comments: str = "") -> bool:
        """Add approval at current step."""
        if self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            step.approver_name = approver_name
            step.status = ActionApprovalStatus.APPROVED.value
            step.responded_at = datetime.utcnow().isoformat()
            step.comments = comments
            
            # Move to next step or complete
            if self.current_step + 1 >= len(self.steps):
                self.status = ActionApprovalStatus.APPROVED.value
                self.completed_at = datetime.utcnow().isoformat()
            else:
                self.current_step += 1
            return True
        return False
    
    def reject(self, approver_name: str, reason: str) -> bool:
        """Reject at current step."""
        if self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            step.approver_name = approver_name
            step.status = ActionApprovalStatus.REJECTED.value
            step.responded_at = datetime.utcnow().isoformat()
            step.comments = reason
            self.status = ActionApprovalStatus.REJECTED.value
            return True
        return False


# =====================================================================
# AUDIT TRAIL
# =====================================================================

@dataclass
class AuditEntry:
    """A single audit trail entry."""
    audit_id: str = field(default_factory=lambda: f"audit_{uuid.uuid4().hex[:8]}")
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    action_type: str = AuditActionType.CREATED.value
    actor: str = ""                    # User/system that performed action
    actor_role: str = ""               # Role of the actor
    target_type: str = "review_card"   # review_card, action, approval
    target_id: str = ""                # ID of target
    details: str = ""                  # Human-readable details
    metadata: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict) -> "AuditEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AuditTrail:
    """Complete audit trail for a review session."""
    trail_id: str = field(default_factory=lambda: f"trail_{uuid.uuid4().hex[:8]}")
    review_card_id: str = ""
    entries: List[AuditEntry] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "trail_id": self.trail_id,
            "review_card_id": self.review_card_id,
            "entries": [e.to_dict() for e in self.entries],
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "AuditTrail":
        entries = [AuditEntry.from_dict(e) for e in data.get("entries", [])]
        return cls(
            trail_id=data.get("trail_id", ""),
            review_card_id=data.get("review_card_id", ""),
            entries=entries,
            created_at=data.get("created_at", ""),
        )
    
    def add_entry(self, entry: AuditEntry) -> None:
        """Add an entry to the trail."""
        self.entries.append(entry)
    
    def get_entries_by_type(self, action_type: str) -> List[AuditEntry]:
        """Get all entries of a specific type."""
        return [e for e in self.entries if e.action_type == action_type]
    
    def get_approval_history(self) -> List[AuditEntry]:
        """Get all approval/rejection entries."""
        return self.get_entries_by_type(AuditActionType.APPROVED.value) + \
               self.get_entries_by_type(AuditActionType.REJECTED.value)
    
    def get_execution_history(self) -> List[AuditEntry]:
        """Get all execution and rollback entries."""
        return self.get_entries_by_type(AuditActionType.EXECUTED.value) + \
               self.get_entries_by_type(AuditActionType.ROLLED_BACK.value)


# =====================================================================
# REVIEW ACTION
# =====================================================================

@dataclass
class ReviewAction:
    """
    A single action with full safety information for review.
    
    Each action includes:
    - Action details (type, description, SQL)
    - Risk assessment (level, indicators, warnings)
    - Approval requirements (workflow, roles)
    - Rollback plan (command, steps, recovery)
    - Operation mode (online/offline)
    """
    # Core identification
    action_id: str = field(default_factory=lambda: f"ra_{uuid.uuid4().hex[:8]}")
    action_type: str = ""              # e.g., "CREATE_INDEX", "VACUUM", "CONFIG_CHANGE"
    title: str = ""
    description: str = ""
    
    # Action content
    sql_command: Optional[str] = None
    config_change: Optional[Dict[str, Any]] = None
    validation_command: Optional[str] = None
    
    # Priority and timing
    priority: str = "medium"           # critical/high/medium/low
    estimated_duration: str = ""       # e.g., "~5 minutes"
    
    # Safety information
    risk: str = ActionRisk.LOW.value
    risk_indicators: RiskIndicators = field(default_factory=RiskIndicators)
    safety_warnings: SafetyWarnings = field(default_factory=SafetyWarnings)
    
    # Operation mode
    operation_mode: str = OperationMode.ONLINE.value  # online/offline/concurrently
    requires_maintenance_window: bool = False
    
    # Approval
    requires_approval: bool = False
    approval_workflow: Optional[ApprovalWorkflow] = None
    
    # Rollback
    rollback_plan: RollbackPlan = field(default_factory=RollbackPlan)
    
    # Impact scope
    impact_scope: str = ImpactScope.TABLE.value
    affected_objects: List[str] = field(default_factory=list)
    
    # State tracking
    status: str = "pending"            # pending/approved/rejected/executed
    executed_at: Optional[str] = None
    executed_by: Optional[str] = None
    
    # References
    kb_entry_id: Optional[str] = None
    references: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "title": self.title,
            "description": self.description,
            "sql_command": self.sql_command,
            "config_change": self.config_change,
            "validation_command": self.validation_command,
            "priority": self.priority,
            "estimated_duration": self.estimated_duration,
            "risk": self.risk,
            "risk_indicators": self.risk_indicators.to_dict(),
            "safety_warnings": self.safety_warnings.to_dict(),
            "operation_mode": self.operation_mode,
            "requires_maintenance_window": self.requires_maintenance_window,
            "requires_approval": self.requires_approval,
            "approval_workflow": self.approval_workflow.to_dict() if self.approval_workflow else None,
            "rollback_plan": self.rollback_plan.to_dict(),
            "impact_scope": self.impact_scope,
            "affected_objects": self.affected_objects,
            "status": self.status,
            "executed_at": self.executed_at,
            "executed_by": self.executed_by,
            "kb_entry_id": self.kb_entry_id,
            "references": self.references,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ReviewAction":
        workflow = None
        if data.get("approval_workflow"):
            workflow = ApprovalWorkflow.from_dict(data["approval_workflow"])
        
        return cls(
            action_id=data.get("action_id", ""),
            action_type=data.get("action_type", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            sql_command=data.get("sql_command"),
            config_change=data.get("config_change"),
            validation_command=data.get("validation_command"),
            priority=data.get("priority", "medium"),
            estimated_duration=data.get("estimated_duration", ""),
            risk=data.get("risk", ActionRisk.LOW.value),
            risk_indicators=RiskIndicators.from_dict(data.get("risk_indicators", {})),
            safety_warnings=SafetyWarnings.from_dict(data.get("safety_warnings", {})),
            operation_mode=data.get("operation_mode", OperationMode.ONLINE.value),
            requires_maintenance_window=data.get("requires_maintenance_window", False),
            requires_approval=data.get("requires_approval", False),
            approval_workflow=workflow,
            rollback_plan=RollbackPlan.from_dict(data.get("rollback_plan", {})),
            impact_scope=data.get("impact_scope", ImpactScope.TABLE.value),
            affected_objects=data.get("affected_objects", []),
            status=data.get("status", "pending"),
            executed_at=data.get("executed_at"),
            executed_by=data.get("executed_by"),
            kb_entry_id=data.get("kb_entry_id"),
            references=data.get("references", []),
        )
    
    def is_safe_to_execute(self) -> bool:
        """Check if action is safe to execute."""
        # Must be approved if required
        if self.requires_approval:
            if not self.approval_workflow or not self.approval_workflow.is_approved():
                return False
        
        # Check for destructive operations without safeguards
        if self.safety_warnings.is_destructive:
            if not self.rollback_plan.rollback_command:
                return False
        
        return True
    
    def needs_approval(self) -> bool:
        """Check if action needs approval."""
        return self.requires_approval or self.risk in (ActionRisk.HIGH.value, ActionRisk.CRITICAL.value)


# =====================================================================
# REVIEW CARD
# =====================================================================

@dataclass
class ReviewCard:
    """
    Complete recommendation card for safety-focused review.
    
    Each card displays:
    - Action details with SQL preview
    - Risk level and visual indicators
    - Impact scope and affected objects
    - Online vs offline operation
    - Approval requirements and workflow
    - Rollback plan
    - Audit trail
    
    Safety Features:
    - Prevents accidental execution of dangerous actions
    - Requires acknowledgment of risks
    - Logs all interactions for audit
    """
    # Core identification
    card_id: str = field(default_factory=lambda: f"rc_{uuid.uuid4().hex[:8]}")
    incident_id: Optional[str] = None
    title: str = ""
    summary: str = ""
    category: str = ""                  # e.g., "query_performance", "index_health"
    severity: str = "medium"            # critical/high/medium/low
    
    # Actions
    actions: List[ReviewAction] = field(default_factory=list)
    
    # Context
    confidence_score: float = 0.5
    evidence_count: int = 0
    root_cause: Optional[str] = None
    
    # Audit
    audit_trail: AuditTrail = field(default_factory=AuditTrail)
    
    # State
    status: str = "pending_review"      # pending_review/in_review/approved/rejected/executed
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "card_id": self.card_id,
            "incident_id": self.incident_id,
            "title": self.title,
            "summary": self.summary,
            "category": self.category,
            "severity": self.severity,
            "actions": [a.to_dict() for a in self.actions],
            "confidence_score": self.confidence_score,
            "evidence_count": self.evidence_count,
            "root_cause": self.root_cause,
            "audit_trail": self.audit_trail.to_dict(),
            "status": self.status,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ReviewCard":
        actions = [ReviewAction.from_dict(a) for a in data.get("actions", [])]
        audit_trail = AuditTrail.from_dict(data.get("audit_trail", {}))
        
        return cls(
            card_id=data.get("card_id", ""),
            incident_id=data.get("incident_id"),
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            category=data.get("category", ""),
            severity=data.get("severity", "medium"),
            actions=actions,
            confidence_score=data.get("confidence_score", 0.5),
            evidence_count=data.get("evidence_count", 0),
            root_cause=data.get("root_cause"),
            audit_trail=audit_trail,
            status=data.get("status", "pending_review"),
            reviewed_by=data.get("reviewed_by"),
            reviewed_at=data.get("reviewed_at"),
            created_at=data.get("created_at", ""),
            expires_at=data.get("expires_at"),
        )
    
    def add_action(self, action: ReviewAction) -> None:
        """Add an action to the card."""
        self.actions.append(action)
        # Log creation
        entry = AuditEntry(
            action_type=AuditActionType.CREATED.value,
            actor="system",
            target_type="action",
            target_id=action.action_id,
            details=f"Action '{action.title}' added to review card",
        )
        self.audit_trail.add_entry(entry)
        # Persist audit entry and card
        try:
            append_audit_entry(self.card_id, entry.to_dict())
            save_card(self.to_dict())
        except Exception:
            pass
    
    def log_view(self, viewer: str) -> None:
        """Log that the card was viewed."""
        entry = AuditEntry(
            action_type=AuditActionType.VIEWED.value,
            actor=viewer,
            target_type="review_card",
            target_id=self.card_id,
            details=f"Review card viewed by {viewer}",
        )
        self.audit_trail.add_entry(entry)
        try:
            append_audit_entry(self.card_id, entry.to_dict())
            save_card(self.to_dict())
        except Exception:
            pass
    
    def log_approval(self, action_id: str, approver: str, comments: str) -> bool:
        """Log action approval."""
        action = self.get_action_by_id(action_id)
        if action and action.approval_workflow:
            if action.approval_workflow.add_approval(approver, comments):
                entry = AuditEntry(
                    action_type=AuditActionType.APPROVED.value,
                    actor=approver,
                    actor_role="approver",
                    target_type="action",
                    target_id=action_id,
                    details=f"Action approved by {approver}: {comments}",
                )
                self.audit_trail.add_entry(entry)
                try:
                    append_audit_entry(self.card_id, entry.to_dict())
                    save_card(self.to_dict())
                except Exception:
                    pass
                return True
        return False
    
    def log_rejection(self, action_id: str, rejector: str, reason: str) -> bool:
        """Log action rejection."""
        action = self.get_action_by_id(action_id)
        if action and action.approval_workflow:
            if action.approval_workflow.reject(rejector, reason):
                entry = AuditEntry(
                    action_type=AuditActionType.REJECTED.value,
                    actor=rejector,
                    actor_role="approver",
                    target_type="action",
                    target_id=action_id,
                    details=f"Action rejected by {rejector}: {reason}",
                )
                self.audit_trail.add_entry(entry)
                try:
                    append_audit_entry(self.card_id, entry.to_dict())
                    save_card(self.to_dict())
                except Exception:
                    pass
                return True
        return False
    
    def log_execution(self, action_id: str, executor: str) -> bool:
        """Log action execution."""
        action = self.get_action_by_id(action_id)
        if action:
            action.status = "executed"
            action.executed_at = datetime.utcnow().isoformat()
            action.executed_by = executor
            
            entry = AuditEntry(
                action_type=AuditActionType.EXECUTED.value,
                actor=executor,
                target_type="action",
                target_id=action_id,
                details=f"Action '{action.title}' executed by {executor}",
            )
            self.audit_trail.add_entry(entry)
            try:
                append_audit_entry(self.card_id, entry.to_dict())
                save_card(self.to_dict())
            except Exception:
                pass
            return True
        return False
    
    def get_action_by_id(self, action_id: str) -> Optional[ReviewAction]:
        """Get action by ID."""
        for action in self.actions:
            if action.action_id == action_id:
                return action
        return None
    
    def get_high_risk_actions(self) -> List[ReviewAction]:
        """Get all high/critical risk actions."""
        return [a for a in self.actions if a.risk in (ActionRisk.HIGH.value, ActionRisk.CRITICAL.value)]
    
    def get_actions_requiring_approval(self) -> List[ReviewAction]:
        """Get all actions requiring approval."""
        return [a for a in self.actions if a.needs_approval()]
    
    def get_offline_actions(self) -> List[ReviewAction]:
        """Get all offline actions."""
        return [a for a in self.actions if a.operation_mode == OperationMode.OFFLINE.value]
    
    def get_audit_summary(self) -> Dict[str, int]:
        """Get audit summary counts."""
        summary = {
            "created": 0,
            "viewed": 0,
            "approved": 0,
            "rejected": 0,
            "executed": 0,
            "rolled_back": 0,
        }
        for entry in self.audit_trail.entries:
            if entry.action_type in summary:
                summary[entry.action_type] += 1
        return summary


# =====================================================================
# REVIEW SESSION
# =====================================================================

@dataclass
class ReviewSession:
    """A complete review session with multiple cards."""
    session_id: str = field(default_factory=lambda: f"session_{uuid.uuid4().hex[:8]}")
    cards: List[ReviewCard] = field(default_factory=list)
    created_by: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "cards": [c.to_dict() for c in self.cards],
            "created_by": self.created_by,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ReviewSession":
        cards = [ReviewCard.from_dict(c) for c in data.get("cards", [])]
        return cls(
            session_id=data.get("session_id", ""),
            cards=cards,
            created_by=data.get("created_by", ""),
            created_at=data.get("created_at", ""),
            completed_at=data.get("completed_at"),
        )
    
    def add_card(self, card: ReviewCard) -> None:
        """Add a card to the session."""
        self.cards.append(card)
    
    def get_pending_actions(self) -> List[ReviewAction]:
        """Get all pending actions across all cards."""
        pending = []
        for card in self.cards:
            for action in card.actions:
                if action.status == "pending":
                    pending.append(action)
        return pending
    
    def get_executed_actions(self) -> List[ReviewAction]:
        """Get all executed actions across all cards."""
        executed = []
        for card in self.cards:
            for action in card.actions:
                if action.status == "executed":
                    executed.append(action)
        return executed


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def create_risk_indicators(risk_level: str) -> RiskIndicators:
    """Create risk indicators based on risk level."""
    risk_config = {
        "low": {
            "badge": "🟢 LOW",
            "color": "#00CC66",
            "summary": "Safe to execute, no approval needed",
        },
        "medium": {
            "badge": "🟡 MEDIUM",
            "color": "#FFCC00",
            "summary": "Standard operation, some caution advised",
        },
        "high": {
            "badge": "🟠 HIGH",
            "color": "#FFAA00",
            "summary": "Significant impact, approval required",
        },
        "critical": {
            "badge": "🔴 CRITICAL",
            "color": "#FF4444",
            "summary": "High risk, multi-level approval required",
        },
    }
    
    config = risk_config.get(risk_level, risk_config["low"])
    return RiskIndicators(
        risk_level=risk_level,
        severity_badge=config["badge"],
        color_code=config["color"],
        risk_summary=config["summary"],
    )


def create_safety_warnings(sql: Optional[str], risk: str) -> SafetyWarnings:
    """Create safety warnings based on SQL and risk level."""
    warnings = SafetyWarnings()
    
    if not sql:
        return warnings
    
    sql_upper = sql.upper()
    
    # Check for destructive operations
    destructive_ops = ["DROP ", "TRUNCATE ", "VACUUM FULL"]
    for op in destructive_ops:
        if op in sql_upper:
            warnings.is_destructive = True
            warnings.warnings.append(f"{op.strip()} detected - this operation may cause data loss")
    
    # Check for index operations
    if "REINDEX" in sql_upper and "CONCURRENTLY" not in sql_upper:
        warnings.is_destructive = True
        warnings.warnings.append("REINDEX without CONCURRENTLY will lock the table")
        warnings.preconditions.append("Ensure no long-running transactions")
    
    # Check for table alterations
    if "ALTER TABLE" in sql_upper:
        warnings.warnings.append("ALTER TABLE may require table rewrite for some operations")
    
    # Check for configuration changes
    if "ALTER SYSTEM" in sql_upper or "postgresql.conf" in sql_upper.lower():
        warnings.warnings.append("Configuration changes may require PostgreSQL restart")
    
    # High risk adds additional warnings
    if risk in ("high", "critical"):
        warnings.warnings.append("This action requires approval before execution")
        warnings.warnings.append("Ensure rollback plan is documented and tested")
    
    return warnings


def create_rollback_plan(action_type: str, sql: Optional[str], risk: str) -> RollbackPlan:
    """Create a rollback plan based on action type and SQL."""
    plan = RollbackPlan()
    
    if not sql:
        return plan
    
    sql_upper = sql.upper()
    
    # Index operations
    if "CREATE INDEX" in sql_upper:
        if "CONCURRENTLY" in sql_upper:
            match_idx = sql_upper.find("INDEX")
            if match_idx != -1:
                # Extract index name (simplified)
                plan.rollback_command = "DROP INDEX CONCURRENTLY {index_name};"
        else:
            plan.rollback_command = "DROP INDEX {index_name}; -- May lock table"
            plan.data_loss_risk = "none"
        plan.recovery_time_estimate = "~30 seconds"
        plan.data_loss_risk = "none"
    
    # VACUUM operations
    elif "VACUUM" in sql_upper:
        plan.rollback_command = "Not applicable - VACUUM only reclaims space"
        plan.recovery_time_estimate = "Instant"
        plan.data_loss_risk = "none"
    
    # ANALYZE operations
    elif "ANALYZE" in sql_upper:
        plan.rollback_command = "Not applicable - ANALYZE only collects statistics"
        plan.recovery_time_estimate = "Instant"
        plan.data_loss_risk = "none"
    
    # Configuration changes
    elif "ALTER SYSTEM" in sql_upper:
        plan.rollback_command = "ALTER SYSTEM SET {parameter} = '{old_value}'; -- Requires restart"
        plan.recovery_time_estimate = "~1 minute + restart time"
        plan.data_loss_risk = "none"
    
    # Destructive operations
    elif any(op in sql_upper for op in ["DROP ", "TRUNCATE"]):
        plan.rollback_command = "Restore from backup - no automatic rollback available"
        plan.recovery_time_estimate = "Varies by backup size"
        plan.data_loss_risk = "major"
        plan.warnings = ["⚠️ Data recovery requires backup restoration"]
    
    # Default rollback
    else:
        plan.rollback_command = "Review PostgreSQL documentation for rollback procedures"
        plan.recovery_time_estimate = "Varies"
        plan.data_loss_risk = "unknown"
    
    return plan


def build_approval_workflow(action_id: str, risk: str, requires_approval: bool) -> Optional[ApprovalWorkflow]:
    """Build approval workflow based on risk level."""
    if not requires_approval and risk not in ("high", "critical"):
        return None
    
    workflow = ApprovalWorkflow(
        action_id=action_id,
        status=ActionApprovalStatus.PENDING.value,
    )
    
    # Define approval steps based on risk
    if risk == "critical":
        workflow.steps = [
            ApprovalStep(step_number=1, approver_role=ApproverRole.DBA_LEAD.value),
            ApprovalStep(step_number=2, approver_role=ApproverRole.SRE_LEAD.value),
            ApprovalStep(step_number=3, approver_role=ApproverRole.MANAGER.value),
        ]
        workflow.requires_multi_approval = True
    elif risk == "high":
        workflow.steps = [
            ApprovalStep(step_number=1, approver_role=ApproverRole.DBA_SENIOR.value),
            ApprovalStep(step_number=2, approver_role=ApproverRole.DBA_LEAD.value),
        ]
        workflow.requires_multi_approval = True
    elif requires_approval:
        workflow.steps = [
            ApprovalStep(step_number=1, approver_role=ApproverRole.DBA_LEAD.value),
        ]
    else:
        return None
    
    return workflow


# =====================================================================
# SERIALIZATION
# =====================================================================

def save_review_session(session: ReviewSession, filepath: str) -> None:
    """Save review session to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(session.to_dict(), f, indent=2)


def load_review_session(filepath: str) -> ReviewSession:
    """Load review session from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return ReviewSession.from_dict(data)


def save_review_card(card: ReviewCard, filepath: str) -> None:
    """Save review card to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(card.to_dict(), f, indent=2)


def load_review_card(filepath: str) -> ReviewCard:
    """Load review card from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return ReviewCard.from_dict(data)

