#!/usr/bin/env python3
"""
review_renderer.py

Renders ReviewCard and ReviewSession to multiple output formats.
Used by examples and tests for displaying recommendation review UI.
"""

import json
from dataclasses import asdict
from typing import Dict, Any, List
from io import StringIO
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from recommendations.review_schema import ReviewCard, ReviewSession, AuditActionType


# Initialize Jinja2 environment for templates
TEMPLATE_DIR = Path(__file__).parent / "templates"
JINJA_ENV = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(['html', 'xml'])
)


def render_to_terminal(card: ReviewCard, use_colors: bool = False) -> str:
    """Render a ReviewCard to ANSI terminal format."""
    output = StringIO()
    output.write(f"\n{'='*60}\n")
    output.write(f"RECOMMENDATION REVIEW CARD\n")
    output.write(f"{'='*60}\n")
    output.write(f"Card ID: {card.card_id}\n")
    output.write(f"Title: {card.title}\n")
    output.write(f"Summary: {card.summary[:80]}...\n" if len(card.summary) > 80 else f"Summary: {card.summary}\n")
    output.write(f"Severity: {card.severity}\n")
    output.write(f"Status: {card.status}\n")
    output.write(f"Confidence: {card.confidence_score:.0%}\n")
    output.write(f"Evidence Count: {card.evidence_count}\n")
    
    if card.actions:
        output.write(f"\nActions ({len(card.actions)}):\n")
        for i, action in enumerate(card.actions, 1):
            output.write(f"  {i}. [{action.risk.upper()}] {action.title}\n")
            if action.sql_command:
                sql = action.sql_command.strip().split('\n')[0]
                output.write(f"     SQL: {sql[:50]}...\n" if len(sql) > 50 else f"     SQL: {sql}\n")
    
    output.write(f"\nAudit Trail: {len(card.audit_trail.entries)} entries\n")
    output.write(f"{'='*60}\n")
    return output.getvalue()


def render_to_markdown(card: ReviewCard) -> str:
    """Render a ReviewCard to Markdown format."""
    output = StringIO()
    output.write(f"# Review Card: {card.title}\n\n")
    output.write(f"**ID:** {card.card_id}\n\n")
    output.write(f"**Summary:** {card.summary}\n\n")
    output.write(f"**Severity:** {card.severity}\n\n")
    output.write(f"**Status:** {card.status}\n\n")
    output.write(f"**Confidence:** {card.confidence_score:.0%}\n\n")
    output.write(f"**Evidence Count:** {card.evidence_count}\n\n")
    
    if card.actions:
        output.write(f"## Actions ({len(card.actions)})\n\n")
        for i, action in enumerate(card.actions, 1):
            output.write(f"### {i}. {action.title}\n\n")
            output.write(f"**Risk:** {action.risk.upper()}\n\n")
            output.write(f"**Description:** {action.description}\n\n")
            if action.sql_command:
                output.write(f"```sql\n{action.sql_command}\n```\n\n")
            if action.safety_warnings.warnings:
                output.write(f"**Warnings:**\n")
                for w in action.safety_warnings.warnings:
                    output.write(f"- {w}\n")
                output.write(f"\n")
    
    output.write(f"\n**Audit Trail:** {len(card.audit_trail.entries)} entries\n")
    return output.getvalue()


def render_to_html(card: ReviewCard) -> str:
    """Render a ReviewCard to HTML format using Jinja2 template."""
    try:
        template = JINJA_ENV.get_template("review_card.html")
        
        # Build context for template
        context = {
            "title": card.title,
            "summary": card.summary,
            "card_id": card.card_id,
            "severity": card.severity,
            "confidence_score": int(card.confidence_score * 100),
            "incident_id": card.incident_id,
            "created_at": card.created_at,
            "root_cause": card.root_cause or "Unknown",
            "evidence_count": card.evidence_count,
            "actions": [],
            "audit_trail": card.audit_trail,
        }
        
        # Build actions context
        high_risk_count = 0
        approval_required_count = 0
        
        for action in card.actions:
            action_dict = {
                "action_id": action.action_id,
                "title": action.title,
                "description": action.description,
                "risk": action.risk,
                "priority": action.priority,
                "sql_command": action.sql_command,
                "operation_mode": action.operation_mode,
                "impact_scope": action.impact_scope,
                "estimated_duration": action.estimated_duration,
                "is_safe_to_execute": action.is_safe_to_execute(),
                "needs_approval": action.needs_approval(),
                "safety_warnings": action.safety_warnings,
                "rollback_plan": action.rollback_plan,
                "approval_workflow": action.approval_workflow,
            }
            context["actions"].append(action_dict)
            
            if action.risk in ("high", "critical"):
                high_risk_count += 1
            if action.needs_approval():
                approval_required_count += 1
        
        context["high_risk_count"] = high_risk_count
        context["approval_required_count"] = approval_required_count
        
        return template.render(**context)
        
    except Exception as e:
        # Fallback to simple HTML if template rendering fails
        return f"""
        <div class="review-card">
            <h2>{card.title}</h2>
            <p><strong>ID:</strong> {card.card_id}</p>
            <p><strong>Summary:</strong> {card.summary}</p>
            <p><strong>Severity:</strong> {card.severity}</p>
            <p><strong>Status:</strong> {card.status}</p>
            <p><strong>Confidence:</strong> {card.confidence_score:.0%}</p>
            <p><strong>Actions:</strong> {len(card.actions)}</p>
            <pre><code>{json.dumps([a.to_dict() for a in card.actions], indent=2)}</code></pre>
        </div>
        """


def render_to_json(card: ReviewCard) -> str:
    """Render a ReviewCard to JSON format."""
    return json.dumps(asdict(card), indent=2, default=str)


def render_session_to_terminal(session: ReviewSession) -> str:
    """Render a ReviewSession to terminal format."""
    output = StringIO()
    output.write(f"\n{'='*60}\n")
    output.write(f"REVIEW SESSION\n")
    output.write(f"{'='*60}\n")
    output.write(f"Session ID: {session.session_id}\n")
    output.write(f"Created: {session.created_at}\n")
    output.write(f"Cards: {len(session.cards)}\n")
    
    for i, card in enumerate(session.cards, 1):
        output.write(f"\n--- Card {i} ---\n")
        output.write(f"ID: {card.card_id}\n")
        output.write(f"Title: {card.title}\n")
        output.write(f"Severity: {card.severity}\n")
        output.write(f"Actions: {len(card.actions)}\n")
    
    output.write(f"{'='*60}\n")
    return output.getvalue()


def render_session_to_markdown(session: ReviewSession) -> str:
    """Render a ReviewSession to Markdown format."""
    output = StringIO()
    output.write(f"# Review Session: {session.session_id}\n\n")
    output.write(f"**Created:** {session.created_at}\n\n")
    output.write(f"**Total Cards:** {len(session.cards)}\n\n")
    
    for i, card in enumerate(session.cards, 1):
        output.write(f"---\n\n")
        output.write(f"## Card {i}: {card.title}\n\n")
        output.write(f"**ID:** {card.card_id}\n\n")
        output.write(render_to_markdown(card))
    
    return output.getvalue()


def render_session_to_html(session: ReviewSession) -> str:
    """Render a ReviewSession to HTML format."""
    output = StringIO()
    output.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Review Session: {session.session_id}</title>
    <link rel="stylesheet" href="../recommendations/static/css/review.css">
</head>
<body>
    <div class="review-container">
        <header class="review-header">
            <h1 class="review-title">Review Session: {session.session_id}</h1>
            <p class="review-summary">Total Cards: {len(session.cards)} | Created: {session.created_at}</p>
        </header>
""")
    
    for card in session.cards:
        output.write(render_to_html(card))
    
    output.write(f"""
    </div>
</body>
</html>
""")
    return output.getvalue()


def render_session_to_json(session: ReviewSession) -> str:
    """Render a ReviewSession to JSON format."""
    return json.dumps(asdict(session), indent=2, default=str)


__all__ = [
    "render_to_terminal",
    "render_to_markdown",
    "render_to_html",
    "render_to_json",
    "render_session_to_terminal",
    "render_session_to_markdown",
    "render_session_to_html",
    "render_session_to_json",
]

