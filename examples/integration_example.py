#!/usr/bin/env python3
"""
Integration Example: Clarification Engine + Incident Renderer

This example demonstrates how to integrate:
1. Clarification Engine - generates questions when confidence is low
2. Incident Renderer - formats the final incident report

Workflow:
- Start with low confidence analysis
- Generate clarification questions
- Process answers
- Render final human-readable report
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from signals.clarification_engine import (
    ClarificationManager,
    EvidenceGapAnalyzer,
    EvidenceType,
)
from reports.incident_renderer import (
    IncidentRenderer,
    RenderFormat,
    RenderContext,
)


def create_sample_incident_with_low_confidence() -> Dict[str, Any]:
    """Create a sample incident with low confidence requiring clarification."""
    return {
        "incident_id": "INC-20260128-002",
        "category": "QUERY_PERFORMANCE",
        "severity": "P3_MEDIUM",
        "status": "INVESTIGATING",
        "timestamp": datetime.utcnow().isoformat(),
        "confidence_score": 0.65,  # Below 0.75 threshold
        "root_causes": [
            {
                "type": "STALE_STATISTICS",
                "description": "Query statistics may be outdated, affecting query planner decisions",
                "confidence": 0.72,
            },
        ],
        "evidence": [
            {
                "evidence_type": "slow_query",
                "value": "SELECT * FROM orders WHERE customer_id = ? taking 2.5s",
                "source": "query_log",
                "confidence": 0.95,
            },
            {
                "evidence_type": "table_stats",
                "value": "Last ANALYZE: 3 days ago",
                "source": "pg_stat_user_tables",
                "confidence": 0.88,
            },
        ],
        "impact": {
            "affected_queries": 15,
            "performance_degradation": "2-3 seconds (200% slower than baseline)",
            "estimated_impact": "~10 users affected",
            "business_context": "Impacts checkout process during peak hours",
        },
        "recommendations": [
            {
                "action": "ANALYZE orders; VACUUM orders;",
                "risk": "LOW",
                "estimated_impact": "Re-optimize query plan",
                "estimated_time": "2-3 minutes",
            },
        ],
        "confidence_breakdown": {
            "base_confidence": 0.72,
            "completeness": 0.80,
            "agreement": 1.0,
            "freshness": 0.90,
        },
        "generated_at": datetime.utcnow().isoformat(),
        "agent_version": "1.0.0",
        "caveats": [
            "Analysis based on limited metrics",
            "Insufficient evidence for high confidence diagnosis",
        ],
    }


def demonstrate_clarification_flow():
    """Demonstrate the clarification question flow."""
    print("\n" + "=" * 80)
    print("CLARIFICATION ENGINE DEMONSTRATION")
    print("=" * 80 + "\n")
    
    # Create low confidence incident
    incident = create_sample_incident_with_low_confidence()
    confidence_score = incident["confidence_score"]
    
    print(f"Initial Confidence Score: {confidence_score:.0%}")
    print(f"Threshold for questions: 0.75 (75%)")
    print(f"Status: Below threshold - questions needed\n")
    
    # Initialize clarification manager
    manager = ClarificationManager(
        confidence_threshold=0.75,
        max_questions=3,
        log_level="INFO"
    )
    
    # Analyze evidence gaps
    analyzer = EvidenceGapAnalyzer()
    root_causes_dict = {
        "likely_causes": [
            {"category": "STATISTICS_MAINTENANCE", "cause_id": "rc_001"},
            {"category": "DEPLOYMENT_SCHEMA", "cause_id": "rc_002"},
        ]
    }
    evidence_dict = {
        "evidence": incident["evidence"]
    }
    
    gaps = analyzer.analyze_gaps(root_causes_dict, evidence_dict)
    
    print(f"Identified {len(gaps)} evidence gaps:\n")
    for gap in gaps[:3]:
        print(f"  • {gap.evidence_type.value}")
        print(f"    Missing count: {gap.missing_count}")
        print(f"    Potential confidence impact: +{gap.potential_confidence_impact:.0%}\n")
    
    # Generate first clarification question
    print("-" * 80)
    print("GENERATED CLARIFICATION QUESTION")
    print("-" * 80 + "\n")
    
    question = manager.generate_question(gaps, confidence_score)
    if question:
        print(f"Question ID: {question.question_id}")
        print(f"Question: {question.question_text}")
        print(f"Category: {question.category.value}")
        print(f"Evidence Type: {question.evidence_type.value}")
        print(f"Priority: {question.priority.name}")
        print(f"\nPossible Answers:")
        for i, option in enumerate(question.answer_options, 1):
            print(f"  {i}. {option}")
        
        # Process a sample answer
        print("\n" + "-" * 80)
        print("SAMPLE ANSWER PROCESSING")
        print("-" * 80 + "\n")
        
        selected_answer = question.answer_options[0]
        print(f"Selected Answer: '{selected_answer}'")
        
        answer = manager.process_answer(question, selected_answer)
        print(f"Confidence Impact: +{answer.confidence_impact:.0%}")
        print(f"Evidence Generated: {answer.evidence_generated}")
        
        # Get updated state
        state = manager.get_state()
        if state:
            new_confidence = state.current_confidence
            print(f"\nUpdated Confidence Score: {new_confidence:.0%}")
            print(f"Improvement: +{(new_confidence - confidence_score):.0%}")
            print(f"Questions Asked: {state.questions_asked_count}/{state.max_questions}")


def demonstrate_incident_rendering():
    """Demonstrate the incident rendering pipeline."""
    print("\n" + "=" * 80)
    print("INCIDENT RENDERING DEMONSTRATION")
    print("=" * 80 + "\n")
    
    # Create high confidence incident with clarifications answered
    incident = {
        "incident_id": "INC-20260128-003",
        "category": "QUERY_PERFORMANCE",
        "severity": "P2_HIGH",
        "status": "IDENTIFIED",
        "timestamp": datetime.utcnow().isoformat(),
        "confidence_score": 0.88,  # After clarifications
        "root_causes": [
            {
                "type": "MISSING_INDEX",
                "description": (
                    "Table 'products' missing index on 'category_id' column. "
                    "Recent deployment added JOIN on this column without creating index."
                ),
                "confidence": 0.92,
            },
            {
                "type": "RECENT_DEPLOYMENT",
                "description": (
                    "New feature deployed 2 hours ago introduced additional JOINs "
                    "in the product listing query."
                ),
                "confidence": 0.85,
            },
        ],
        "evidence": [
            {
                "evidence_type": "sequential_scan_detected",
                "value": "Sequential scan on products (500K rows) taking 4.2s",
                "source": "explain_analyze",
                "confidence": 0.96,
            },
            {
                "evidence_type": "missing_index",
                "value": "category_id frequently used in WHERE and JOIN conditions",
                "source": "query_analysis",
                "confidence": 0.94,
            },
            {
                "evidence_type": "deployment",
                "value": "Deployment INC-8442 at 2026-01-28 01:15 UTC introduced product_details join",
                "source": "deployment_log",
                "confidence": 0.99,
            },
            {
                "evidence_type": "plan_change",
                "value": "Query plan changed from nested loop to hash join after deployment",
                "source": "pg_stat_statements",
                "confidence": 0.91,
            },
        ],
        "impact": {
            "affected_queries": 125,
            "performance_degradation": "4-5 seconds (400% slower than baseline)",
            "estimated_impact": "~300 users during peak, impacting checkout process",
            "business_context": "Critical - blocks e-commerce transactions during peak hours",
        },
        "recommendations": [
            {
                "action": "CREATE INDEX idx_products_category_id ON products(category_id)",
                "risk": "LOW",
                "estimated_impact": "Expected 95% improvement in query time",
                "estimated_time": "< 1 minute",
            },
            {
                "action": "ANALYZE products;",
                "risk": "LOW",
                "estimated_impact": "Update statistics with new index",
                "estimated_time": "30 seconds",
            },
            {
                "action": "Verify query plan after index creation",
                "risk": "LOW",
                "estimated_impact": "Ensure optimizer uses new index",
                "estimated_time": "5 minutes",
            },
        ],
        "confidence_breakdown": {
            "base_confidence": 0.90,
            "completeness": 0.95,
            "agreement": 1.15,
            "freshness": 0.99,
            "conflict_penalty": 1.0,
        },
        "clarification_questions": [
            {
                "question_text": "Was there a recent deployment (application or database schema change) in the last 24 hours?",
                "answer": "Yes - within last 6 hours",
            },
            {
                "question_text": "When was the query or incident last performing normally?",
                "answer": "Within last hour (before deployment)",
            },
        ],
        "generated_at": datetime.utcnow().isoformat(),
        "agent_version": "1.0.0",
        "caveats": [
            "Recommendation assumes no other queries depend on old plan",
            "Test in non-production environment first if possible",
        ],
    }
    
    # Initialize renderer
    renderer = IncidentRenderer()
    
    # Render in different formats
    formats = [
        (RenderFormat.TERMINAL, "Terminal/Console"),
        (RenderFormat.MARKDOWN, "Markdown"),
        (RenderFormat.SLACK, "Slack"),
    ]
    
    for fmt, name in formats:
        print(f"\n{'-' * 80}")
        print(f"{name.upper()} FORMAT")
        print(f"{'-' * 80}\n")
        
        context = RenderContext(
            format=fmt,
            include_evidence_details=(fmt == RenderFormat.MARKDOWN),
            max_evidence_items=3,
            max_recommendations=2,
        )
        
        output = renderer.render(incident, context)
        
        # Show first 500 chars for terminal (to keep output manageable)
        if fmt == RenderFormat.TERMINAL:
            print(output[:800] + "\n[...output truncated for brevity...]")
        else:
            lines = output.split("\n")
            print("\n".join(lines[:15]))
            if len(lines) > 15:
                print(f"\n[...{len(lines) - 15} more lines...]")


def demonstrate_end_to_end_flow():
    """Demonstrate complete end-to-end flow."""
    print("\n" + "=" * 80)
    print("END-TO-END WORKFLOW: LOW CONFIDENCE → CLARIFICATION → RENDERING")
    print("=" * 80 + "\n")
    
    # Step 1: Low confidence analysis
    print("Step 1: Initial Analysis")
    print("-" * 80)
    incident = create_sample_incident_with_low_confidence()
    print(f"Incident ID: {incident['incident_id']}")
    print(f"Initial Confidence: {incident['confidence_score']:.0%}")
    print(f"Status: {incident['status']}")
    print(f"Evidence Items: {len(incident['evidence'])}")
    
    # Step 2: Generate clarifications
    print("\n\nStep 2: Generate Clarification Questions")
    print("-" * 80)
    
    manager = ClarificationManager(max_questions=2)
    analyzer = EvidenceGapAnalyzer()
    
    root_causes = {
        "likely_causes": [
            {"category": "STATISTICS_MAINTENANCE", "cause_id": "rc_001"},
        ]
    }
    evidence = {"evidence": incident["evidence"]}
    
    gaps = analyzer.analyze_gaps(root_causes, evidence)
    
    if gaps:
        question = manager.generate_question(gaps, incident["confidence_score"])
        if question:
            print(f"Generated Question: {question.question_text}")
            print(f"Options: {', '.join(question.answer_options[:2])}")
            
            # Step 3: Process answer
            print("\n\nStep 3: Process Answer & Update Confidence")
            print("-" * 80)
            
            answer = manager.process_answer(
                question,
                "Yes, within last 24 hours"
            )
            
            state = manager.get_state()
            updated_confidence = state.current_confidence if state else incident["confidence_score"]
            
            print(f"Answer: 'Yes, within last 24 hours'")
            print(f"Confidence Impact: +{answer.confidence_impact:.0%}")
            print(f"New Confidence: {updated_confidence:.0%}")
            
            # Update incident with new confidence
            incident["confidence_score"] = updated_confidence
            incident["status"] = "IDENTIFIED" if updated_confidence >= 0.75 else "INVESTIGATING"
    
    # Step 4: Render final report
    print("\n\nStep 4: Render Final Incident Report (Terminal Format)")
    print("-" * 80)
    
    renderer = IncidentRenderer()
    context = RenderContext(
        format=RenderFormat.TERMINAL,
        colors_enabled=True,
        max_evidence_items=4,
        max_recommendations=2,
    )
    
    output = renderer.render(incident, context)
    print(output[:1000] + "\n[...output truncated...]")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_clarification_flow()
    demonstrate_incident_rendering()
    demonstrate_end_to_end_flow()
    
    print("\n" + "=" * 80)
    print("✅ INTEGRATION DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  1. Clarification Engine - Dynamic question generation")
    print("  2. Confidence Tracking - Updates as answers are processed")
    print("  3. Evidence Gap Analysis - Identifies missing critical evidence")
    print("  4. Incident Rendering - Multi-format output (Terminal, Markdown, Slack)")
    print("  5. Executive Summary - Professional incident reports")
