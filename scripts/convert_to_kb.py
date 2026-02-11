#!/usr/bin/env python3
"""
PostgreSQL Knowledge Base Conversion Script

Converts the 6 log JSON files to the unified KB JSON schema format
optimized for Llama model training.

Usage:
    python scripts/convert_to_kb.py [--output data/kb_unified.json]
"""

import json
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


# Define schema version
KB_VERSION = "1.0.0"

# Source file mapping
SOURCE_FILES = {
    "query_matric": "data/logs/query_matric.json",
    "index_health": "data/logs/index_health.json",
    "locking_analysis": "data/logs/locking_analysis.json",
    "maintenance_checks": "data/logs/maintenance_checks.json",
    "configuration_parameters": "data/logs/configuration_parameters.json",
    "application_impact": "data/logs/application_impact.json",
    "add_index_fix": "data/logs/add_index_fix.json"
}

# Category mapping based on source file
CATEGORY_MAP = {
    "query_matric": "query_performance",
    "index_health": "index_health",
    "locking_analysis": "locking",
    "maintenance_checks": "maintenance",
    "configuration_parameters": "configuration",
    "application_impact": "application_impact"
}

# Severity mapping
SEVERITY_MAP = {
    "critical": "critical",
    "P1": "critical",
    "high": "high",
    "P2": "high",
    "medium": "medium",
    "P3": "medium",
    "low": "low",
    "info": "info"
}


def extract_metadata(source_name: str, source_data: Dict) -> Dict:
    """Extract metadata from source data."""
    metadata = {
        "kb_id": f"kb_{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "category": CATEGORY_MAP.get(source_name, "general"),
        "severity": SEVERITY_MAP.get(
            source_data.get("metadata", {}).get("severity", "medium"),
            "medium"
        ),
        "source": source_data.get("metadata", {}).get("source", "unknown"),
        "version": source_data.get("metadata", {}).get("version", KB_VERSION),
        "created_at": source_data.get("metadata", {}).get("created_at", datetime.now().isoformat() + "Z"),
        "tags": [source_name, CATEGORY_MAP.get(source_name, "general")]
    }
    
    # Add tags based on content
    if "recommendations" in source_data:
        if source_data["recommendations"].get("immediate_actions"):
            metadata["tags"].append("immediate_action_available")
    if "detection_signals" in source_data:
        if source_data["detection_signals"].get("anomaly_flags"):
            metadata["tags"].append("anomaly_detected")
    
    return metadata


def extract_problem_identity(source_data: Dict) -> Dict:
    """Extract problem identity from source data."""
    problem = {
        "issue_type": source_data.get("problem_identity", {}).get("issue_type", "Unknown Issue"),
        "short_description": source_data.get("problem_identity", {}).get(
            "short_description", 
            source_data.get("alert", {}).get("subject", "No description available")
        ),
        "symptoms": source_data.get("problem_identity", {}).get("symptoms", []),
        "affected_components": source_data.get("problem_identity", {}).get("affected_components", [])
    }
    
    # Add description from alert if available
    if "alert" in source_data:
        alert_body = source_data.get("alert", {}).get("alert_body", {})
        problem["long_description"] = alert_body.get("impact_summary", "")
    
    return problem


def extract_detection_signals(source_data: Dict, source_name: str) -> Dict:
    """Extract detection signals from source data."""
    signals = {}
    
    # Extract metrics
    if "detection_signals" in source_data:
        signals["metrics"] = source_data["detection_signals"].get("metrics", {})
        signals["thresholds"] = source_data["detection_signals"].get("thresholds", {})
        signals["anomaly_flags"] = source_data["detection_signals"].get("anomaly_flags", {})
        signals["detection_method"] = source_data["detection_signals"].get("detection_method", "threshold_alert")
    elif "query_metrics" in source_data:
        signals["metrics"] = source_data.get("query_metrics", {}).get("metrics", {})
        signals["anomaly_flags"] = {
            "execution_time_anomaly": True,
            "query_detected": True
        }
        signals["detection_method"] = "pg_stat_statements_analysis"
    elif "hardware_capacity" in source_data:
        signals["metrics"] = source_data.get("hardware_capacity", {})
        signals["anomaly_flags"] = {
            "io_wait_detected": source_data.get("hardware_capacity", {}).get("disk_io_wait_percent", 0) > 20
        }
    
    # Add source queries based on source type
    source_queries = []
    if source_name == "query_matric":
        source_queries = [
            "SELECT query, calls, mean_time, total_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 20"
        ]
    elif source_name == "index_health":
        source_queries = [
            "SELECT indexrelname, idx_scan, idx_tup_read, idx_tup_fetch, pg_relation_size(indexrelid) FROM pg_stat_user_indexes"
        ]
    elif source_name == "locking_analysis":
        source_queries = [
            "SELECT pid, usename, application_name, state, wait_event, query FROM pg_stat_activity WHERE wait_event IS NOT NULL"
        ]
    elif source_name == "maintenance_checks":
        source_queries = [
            "SELECT schemaname, tablename, n_dead_tup, n_live_tup, last_vacuum, last_analyze FROM pg_stat_user_tables"
        ]
    
    signals["source_queries"] = source_queries
    
    return signals


def extract_context(source_data: Dict) -> Dict:
    """Extract context information from source data."""
    context = {}
    
    # Query fingerprint
    if "context" in source_data:
        context["query_fingerprint"] = source_data["context"].get("query_fingerprint", {})
        context["environment"] = source_data["context"].get("environment", {})
    
    # Environment from application_impact
    if "alert" in source_data:
        alert_body = source_data.get("alert", {}).get("alert_body", {})
        context["environment"] = {
            "database": alert_body.get("database", "unknown"),
            "schema": alert_body.get("schema", "public")
        }
        context["tables_involved"] = alert_body.get("tables_involved", [])
    
    # Tables from table_statistics
    if "table_statistics" in source_data:
        tables = source_data["table_statistics"].get("tables", [])
        context["tables_involved"] = [t.get("table_name", "") for t in tables if t.get("table_name")]
    
    return context


def extract_root_cause_analysis(source_data: Dict) -> Dict:
    """Extract root cause analysis from source data."""
    rca = {}
    
    if "root_cause_analysis" in source_data:
        rca["primary_cause"] = source_data["root_cause_analysis"].get(
            "primary_cause",
            source_data["root_cause_analysis"].get("primary_category", "Unknown")
        )
        rca["contributing_factors"] = source_data["root_cause_analysis"].get(
            "secondary_factors",
            source_data["root_cause_analysis"].get("contributing_factors", [])
        )
    elif "problem_identity" in source_data:
        rca["primary_cause"] = source_data["problem_identity"].get(
            "root_cause",
            source_data.get("problem_identity", {}).get("short_description", "Unknown")
        )
        rca["contributing_factors"] = []
    
    return rca


def extract_recommendations(source_data: Dict, source_name: str) -> Dict:
    """Extract recommendations from source data."""
    recs = {
        "immediate_actions": [],
        "long_term_fixes": [],
        "validation_steps": [],
        "preventive_actions": []
    }
    
    # Extract from existing recommendations section
    if "recommendations" in source_data:
        source_recs = source_data["recommendations"]
        recs["immediate_actions"] = source_recs.get("immediate_actions", [])
        recs["long_term_fixes"] = source_recs.get("long_term_fixes", [])
        recs["validation_steps"] = source_recs.get("validation_steps", [])
        recs["preventive_actions"] = source_recs.get("preventive_actions", [])
    
    # Extract from resolution section
    if "resolution" in source_data:
        resolution = source_data["resolution"]
        recs["immediate_actions"].extend([
            {"action": a, "risk": "low"} 
            for a in resolution.get("immediate_fix", [])
        ])
        recs["long_term_fixes"].extend([
            {"action": a, "priority": "medium"}
            for a in resolution.get("permanent_fix", [])
        ])
        recs["preventive_actions"].extend(resolution.get("preventive_actions", []))
    
    # Extract from maintenance_checks
    if "maintenance_checks" in source_data:
        maintenance = source_data["maintenance_checks"]
        if "actions" in maintenance:
            recs["immediate_actions"].extend([
                {"action": a, "risk": "low"} 
                for a in maintenance["actions"]
            ])
        if "recommended_actions" in maintenance:
            recs["long_term_fixes"].extend([
                {"action": a, "priority": "medium"}
                for a in maintenance["recommended_actions"]
            ])
    
    # Generate source-specific recommendations
    if source_name == "query_matric":
        recs["validation_steps"].extend([
            "Compare new execution plan with baseline",
            "Verify index usage",
            "Confirm temp file creation stopped",
            "Monitor P95 latency for 24h"
        ])
    elif source_name == "index_health":
        recs["long_term_fixes"].extend([
            {"action": "Run REINDEX to remove index bloat", "sql_example": "REINDEX INDEX index_name;", "priority": "high"},
            {"action": "Consider using REINDEX CONCURRENTLY for zero-downtime reindexing", "priority": "medium"}
        ])
    elif source_name == "locking_analysis":
        recs["immediate_actions"].extend([
            {"action": "Identify and analyze blocking transaction", "risk": "medium"},
            {"action": "Consider terminating long-running blocking query", "sql_example": "SELECT pg_terminate_backend(pid);", "risk": "high"}
        ])
    elif source_name == "maintenance_checks":
        recs["immediate_actions"].extend([
            {"action": "Run VACUUM ANALYZE on affected tables", "sql_example": "VACUUM (VERBOSE) ANALYZE table_name;", "risk": "low"}
        ])
        recs["long_term_fixes"].extend([
            {"action": "Tune autovacuum settings", "config_example": "ALTER SYSTEM SET autovacuum_max_workers = 4;", "priority": "high"},
            {"action": "Schedule regular VACUUM ANALYZE during maintenance window", "priority": "medium"}
        ])
    
    return recs


def extract_evidence(source_data: Dict, source_name: str) -> Dict:
    """Extract evidence from source data."""
    evidence = {}
    
    if source_name == "query_matric":
        evidence["query_metrics"] = source_data.get("detection_signals", {}).get("metrics", {})
    elif source_name == "application_impact":
        evidence["query_metrics"] = source_data.get("query_metrics", {}).get("metrics", {})
        evidence["table_statistics"] = source_data.get("table_statistics", {}).get("tables", [])
        evidence["index_statistics"] = source_data.get("index_health", {}).get("indexes", [])
        evidence["locking_details"] = source_data.get("locking_analysis", {})
        evidence["configuration_values"] = source_data.get("configuration_parameters", {})
        evidence["hardware_metrics"] = source_data.get("hardware_capacity", {})
    else:
        # Generic evidence extraction
        evidence = {k: v for k, v in source_data.items() 
                   if k not in ["metadata", "problem_identity", "detection_signals", 
                               "context", "root_cause_analysis", "impact_analysis", 
                               "recommendations", "confidence"]}
    
    return evidence


def extract_confidence(source_data: Dict) -> Dict:
    """Extract confidence metrics from source data."""
    if "confidence" in source_data:
        return source_data["confidence"]
    
    return {
        "confidence_score": 0.85,
        "confidence_reasoning": "Analysis based on PostgreSQL statistics and metrics",
        "evidence_count": 1
    }


def convert_source_to_kb_entry(source_name: str, source_path: str) -> Optional[Dict]:
    """Convert a single source file to KB entry format."""
    try:
        with open(source_path, 'r') as f:
            source_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Source file not found: {source_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in {source_path}: {e}")
        return None
    
    entry = {
        "metadata": extract_metadata(source_name, source_data),
        "problem_identity": extract_problem_identity(source_data),
        "detection_signals": extract_detection_signals(source_data, source_name),
        "context": extract_context(source_data),
        "root_cause_analysis": extract_root_cause_analysis(source_data),
        "recommendations": extract_recommendations(source_data, source_name),
        "evidence": extract_evidence(source_data, source_name),
        "confidence": extract_confidence(source_data)
    }
    
    # Add impact analysis if available
    if "impact_analysis" in source_data:
        entry["impact_analysis"] = source_data["impact_analysis"]
    elif "application_impact" in source_data:
        entry["impact_analysis"] = {
            "application_impact": source_data.get("application_impact", {})
        }
    
    return entry


def convert_all_sources(output_path: str) -> Dict:
    """Convert all source files and create unified KB."""
    entries = []
    
    for source_name, source_path in SOURCE_FILES.items():
        entry = convert_source_to_kb_entry(source_name, source_path)
        if entry:
            entries.append(entry)
            print(f"Converted: {source_name} -> {source_path}")
    
    kb = {
        "kb_version": KB_VERSION,
        "generated_at": datetime.now().isoformat() + "Z",
        "entry_count": len(entries),
        "entries": entries
    }
    
    # Write output
    with open(output_path, 'w') as f:
        json.dump(kb, f, indent=2)
    
    print(f"\nUnified KB written to: {output_path}")
    print(f"Total entries: {len(entries)}")
    
    return kb


def main():
    parser = argparse.ArgumentParser(
        description="Convert PostgreSQL log files to KB JSON schema"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/kb_unified.json",
        help="Output path for unified KB file"
    )
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run conversion
    kb = convert_all_sources(args.output)
    
    print("\nConversion complete!")
    print(f"Schema version: {kb['kb_version']}")
    print(f"Entries generated: {kb['entry_count']}")
    
    return kb


if __name__ == "__main__":
    main()

