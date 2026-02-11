#!/usr/bin/env python3
"""
Security Recommender Module

Generate DBA-safe security recommendations for PostgreSQL.
Designed for compliance and security hardening with proper safety measures.

Features:
- Risk assessment for security issues
- Approval workflow for critical changes
- Rollback procedures
- Compliance mapping (PCI-DSS, GDPR, SOC2)
- Phased implementation guidance
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from signals.security_signals import SecurityReport, SecurityCheck
from signals.signal_engine import SignalResult, Signal


# -------------------------------------------------------------------
# Data Models
# -------------------------------------------------------------------

@dataclass
class SecurityAction:
    """
    A single security action with full safety information.
    """
    action: str
    sql: Optional[str]
    risk_level: str  # low, medium, high, critical
    is_online: bool  # Can run while database is active
    requires_approval: bool
    estimated_downtime: str
    rollback_notes: str
    priority: str  # immediate, soon, scheduled
    compliance_frameworks: List[str]
    verification_command: Optional[str]


@dataclass
class SecurityRecommendation:
    """
    A complete security recommendation with actions.
    """
    check_name: str
    category: str
    severity: str
    title: str
    description: str
    current_state: str
    desired_state: str
    actions: List[SecurityAction]
    confidence: float
    references: List[str]
    compliance_tags: List[str]


@dataclass
class SecurityRecommendationReport:
    """
    Complete security recommendation report.
    """
    recommendations: List[SecurityRecommendation]
    overall_score: float
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    compliance_summary: Dict[str, Dict[str, Any]]
    risk_summary: Dict[str, List[str]]
    implementation_plan: List[Dict[str, Any]]
    timestamp: str


# -------------------------------------------------------------------
# Security Recommender
# -------------------------------------------------------------------

class SecurityRecommender:
    """
    Generate DBA-safe security recommendations.
    
    Maps security signals to actionable recommendations with:
    - Proper risk assessment
    - Approval requirements
    - Rollback procedures
    - Compliance mapping
    """

    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

    def recommend(
        self,
        signal_result: SignalResult,
        security_report: Optional[SecurityReport] = None,
    ) -> SecurityRecommendationReport:
        """
        Generate security recommendations.
        
        Args:
            signal_result: SignalResult from SecuritySignalGenerator
            security_report: Optional SecurityReport for detailed analysis
            
        Returns:
            SecurityRecommendationReport with all recommendations
        """
        recommendations: List[SecurityRecommendation] = []
        
        # Process each signal
        for signal in signal_result.signals:
            rec = self._signal_to_recommendation(signal)
            if rec:
                recommendations.append(rec)

        # Calculate counts
        critical = sum(1 for r in recommendations if r.severity == "critical")
        high = sum(1 for r in recommendations if r.severity == "high")
        medium = sum(1 for r in recommendations if r.severity == "medium")
        low = sum(1 for r in recommendations if r.severity == "low")

        # Calculate overall score
        total_checks = len(recommendations)
        if total_checks > 0:
            score = sum(
                4 - {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(r.severity, 0)
                for r in recommendations
            )
            max_score = total_checks * 4
            overall_score = (score / max_score) * 100
        else:
            overall_score = 100.0

        # Build compliance summary
        compliance_summary = self._build_compliance_summary(recommendations)

        # Build risk summary
        risk_summary = {
            "critical": [r.title for r in recommendations if r.severity == "critical"],
            "high": [r.title for r in recommendations if r.severity == "high"],
            "medium": [r.title for r in recommendations if r.severity == "medium"],
            "low": [r.title for r in recommendations if r.severity == "low"],
        }

        # Build implementation plan
        implementation_plan = self._build_implementation_plan(recommendations)

        return SecurityRecommendationReport(
            recommendations=recommendations,
            overall_score=overall_score,
            critical_count=critical,
            high_count=high,
            medium_count=medium,
            low_count=low,
            compliance_summary=compliance_summary,
            risk_summary=risk_summary,
            implementation_plan=implementation_plan,
            timestamp=datetime.utcnow().isoformat(),
        )

    def _signal_to_recommendation(self, signal: Signal) -> Optional[SecurityRecommendation]:
        """Convert a security signal to a recommendation."""
        signal_map = {
            "ssl_not_enabled": self._recommend_ssl_enable,
            "ssl_preload_not_configured": self._recommend_ssl_preload,
            "listen_addresses_vulnerable": self._recommend_listen_addresses,
            "md5_password_encryption": self._recommend_password_encryption,
            "weak_authentication_method": self._recommend_auth_method,
            "audit_logging_disabled": self._recommend_audit_logging,
            "excessive_max_connections": self._recommend_connection_limit,
            "excessive_superuser_accounts": self._recommend_superuser_reduction,
            "row_level_security_disabled": self._recommend_rls_enable,
            "multiple_failed_authentications": self._recommend_failed_logins,
            "connection_from_unknown_host": self._recommend_connection_review,
            "potential_sql_injection": self._recommend_sql_injection,
            "ssl_connection_errors": self._recommend_ssl_review,
            "role_modification_detected": self._recommend_audit_role_changes,
        }

        handler = signal_map.get(signal.name)
        if handler:
            return handler(signal)

        return None

    def _recommend_ssl_enable(self, signal: Signal) -> SecurityRecommendation:
        """Recommendation for enabling SSL."""
        return SecurityRecommendation(
            check_name=signal.name,
            category="encryption",
            severity=signal.severity,
            title="Enable SSL/TLS for Database Connections",
            description="SSL/TLS encryption is required to protect data in transit.",
            current_state="SSL is currently disabled",
            desired_state="SSL enabled with valid certificates",
            actions=[
                SecurityAction(
                    action="Generate SSL certificate",
                    sql="openssl req -new -x509 -days 365 -nodes -text -out server.crt -keyout server.key -subj \"/CN=postgres\"",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Remove ssl cert files and set ssl=off",
                    priority="immediate",
                    compliance_frameworks=["pci_dss", "gdpr", "soc2"],
                    verification_command="SHOW ssl;",
                ),
                SecurityAction(
                    action="Configure SSL in postgresql.conf",
                    sql="ALTER SYSTEM SET ssl = on; SELECT pg_reload_conf();",
                    risk_level="medium",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none (requires reload)",
                    rollback_notes="ALTER SYSTEM SET ssl = off; SELECT pg_reload_conf();",
                    priority="immediate",
                    compliance_frameworks=["pci_dss", "gdpr", "soc2"],
                    verification_command="SHOW ssl;",
                ),
                SecurityAction(
                    action="Update pg_hba.conf for SSL enforcement",
                    sql=None,  # Manual configuration needed
                    risk_level="high",
                    is_online=False,
                    requires_approval=True,
                    estimated_downtime="brief (connection restart)",
                    rollback_notes="Restore original pg_hba.conf",
                    priority="soon",
                    compliance_frameworks=["pci_dss"],
                    verification_command="SELECT ssl from pg_stat_ssl WHERE pid = pg_backend_pid();",
                ),
            ],
            confidence=signal.confidence,
            references=[
                "https://www.postgresql.org/docs/current/ssl-tcp.html",
                "https://www.postgresql.org/docs/current/auth-pg-hba-conf.html",
            ],
            compliance_tags=["pci_dss", "gdpr", "soc2"],
        )

    def _recommend_ssl_preload(self, signal: Signal) -> SecurityRecommendation:
        """Recommendation for SSL preload configuration."""
        return SecurityRecommendation(
            check_name=signal.name,
            category="encryption",
            severity=signal.severity,
            title="Configure SSL Library Preloading",
            description="Preloading the SSL library enables SSL session caching for better performance.",
            current_state="SSL library not preloaded",
            desired_state="ssl in shared_preload_libraries",
            actions=[
                SecurityAction(
                    action="Add ssl to shared_preload_libraries",
                    sql="ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements,ssl'; SELECT pg_reload_conf();",
                    risk_level="medium",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="brief (requires reload)",
                    rollback_notes="ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';",
                    priority="scheduled",
                    compliance_frameworks=[],
                    verification_command="SHOW shared_preload_libraries;",
                ),
            ],
            confidence=signal.confidence,
            references=["https://www.postgresql.org/docs/current/ssl-session-caching.html"],
            compliance_tags=[],
        )

    def _recommend_listen_addresses(self, signal: Signal) -> SecurityRecommendation:
        """Recommendation for restricting listen addresses."""
        listen_val = signal.data.get("listen_addresses", "*")
        return SecurityRecommendation(
            check_name=signal.name,
            category="network_security",
            severity=signal.severity,
            title="Restrict Database Listen Addresses",
            description=f"Database currently listens on '{listen_val}' - should only listen on required interfaces.",
            current_state=f"listen_addresses = '{listen_val}'",
            desired_state="listen_addresses = '127.0.0.1' or specific internal IPs",
            actions=[
                SecurityAction(
                    action="Restrict listen_addresses to localhost",
                    sql="ALTER SYSTEM SET listen_addresses = '127.0.0.1'; SELECT pg_reload_conf();",
                    risk_level="high",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="brief (requires reload)",
                    rollback_notes="ALTER SYSTEM SET listen_addresses = '*';",
                    priority="soon",
                    compliance_frameworks=["pci_dss"],
                    verification_command="SHOW listen_addresses;",
                ),
                SecurityAction(
                    action="Configure firewall rules",
                    sql=None,
                    risk_level="low",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none",
                    rollback_notes="Remove firewall rules",
                    priority="soon",
                    compliance_frameworks=["pci_dss"],
                    verification_command="iptables -L -n | grep 5432",
                ),
            ],
            confidence=signal.confidence,
            references=["https://www.postgresql.org/docs/current/runtime-config-connection.html"],
            compliance_tags=["pci_dss"],
        )

    def _recommend_password_encryption(self, signal: Signal) -> SecurityRecommendation:
        """Recommendation for upgrading password encryption."""
        return SecurityRecommendation(
            check_name=signal.name,
            category="authentication",
            severity=signal.severity,
            title="Upgrade Password Encryption to SCRAM-SHA-256",
            description="MD5 password encryption is considered weak and should be upgraded to SCRAM-SHA-256.",
            current_state="password_encryption = md5",
            desired_state="password_encryption = scram-sha-256",
            actions=[
                SecurityAction(
                    action="Update password_encryption setting",
                    sql="ALTER SYSTEM SET password_encryption = 'scram-sha-256'; SELECT pg_reload_conf();",
                    risk_level="medium",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none (requires reload)",
                    rollback_notes="ALTER SYSTEM SET password_encryption = 'md5';",
                    priority="soon",
                    compliance_frameworks=["pci_dss"],
                    verification_command="SHOW password_encryption;",
                ),
                SecurityAction(
                    action="Reset existing password hashes",
                    sql="ALTER USER username WITH PASSWORD 'new_password';",
                    risk_level="medium",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Users will need to reset passwords",
                    priority="scheduled",
                    compliance_frameworks=["pci_dss"],
                    verification_command="SELECT rolname, rolpassword LIKE 'SCRAM%' FROM pg_authid;",
                ),
            ],
            confidence=signal.confidence,
            references=[
                "https://www.postgresql.org/docs/current/auth-password.html",
                "https://www.postgresql.org/docs/current/auth-methods.html#AUTH-PASSWORD",
            ],
            compliance_tags=["pci_dss"],
        )

    def _recommend_auth_method(self, signal: Signal) -> SecurityRecommendation:
        """Recommendation for weak authentication methods."""
        return SecurityRecommendation(
            check_name=signal.name,
            category="authentication",
            severity=signal.severity,
            title="Replace Weak Authentication Methods",
            description="Using 'trust' or 'ident' authentication can allow unauthorized access.",
            current_state="Using weak authentication method",
            desired_state="scram-sha-256 or certificate authentication",
            actions=[
                SecurityAction(
                    action="Review pg_hba.conf entries",
                    sql="SELECT * FROM pg_hba_file_rules();",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not applicable - read operation",
                    priority="immediate",
                    compliance_frameworks=["pci_dss", "soc2"],
                    verification_command="SELECT * FROM pg_hba_file_rules() WHERE auth_method NOT IN ('scram-sha-256', 'cert', 'md5');",
                ),
                SecurityAction(
                    action="Update pg_hba.conf to use strong authentication",
                    sql=None,  # Manual configuration
                    risk_level="critical",
                    is_online=False,
                    requires_approval=True,
                    estimated_downtime="connection interruption",
                    rollback_notes="Restore from backup pg_hba.conf",
                    priority="immediate",
                    compliance_frameworks=["pci_dss", "soc2"],
                    verification_command="SELECT DISTINCT auth_method FROM pg_hba_file_rules();",
                ),
            ],
            confidence=signal.confidence,
            references=[
                "https://www.postgresql.org/docs/current/auth-pg-hba-conf.html",
                "https://www.postgresql.org/docs/current/auth-methods.html",
            ],
            compliance_tags=["pci_dss", "soc2"],
        )

    def _recommend_audit_logging(self, signal: Signal) -> SecurityRecommendation:
        """Recommendation for enabling audit logging."""
        return SecurityRecommendation(
            check_name=signal.name,
            category="audit_logging",
            severity=signal.severity,
            title="Enable Comprehensive Audit Logging",
            description="Audit logging is not properly configured for security compliance.",
            current_state="Audit logging incomplete or disabled",
            desired_state="All connections, disconnections, and statements logged with pgaudit",
            actions=[
                SecurityAction(
                    action="Enable connection logging",
                    sql="ALTER SYSTEM SET log_connections = on; SELECT pg_reload_conf();",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="ALTER SYSTEM SET log_connections = off;",
                    priority="soon",
                    compliance_frameworks=["pci_dss", "gdpr", "soc2"],
                    verification_command="SHOW log_connections;",
                ),
                SecurityAction(
                    action="Enable disconnection logging",
                    sql="ALTER SYSTEM SET log_disconnections = on; SELECT pg_reload_conf();",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="ALTER SYSTEM SET log_disconnections = off;",
                    priority="soon",
                    compliance_frameworks=["pci_dss", "gdpr", "soc2"],
                    verification_command="SHOW log_disconnections;",
                ),
                SecurityAction(
                    action="Set appropriate log_statement",
                    sql="ALTER SYSTEM SET log_statement = 'ddl'; SELECT pg_reload_conf();",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="ALTER SYSTEM SET log_statement = 'none';",
                    priority="soon",
                    compliance_frameworks=["pci_dss", "gdpr", "soc2"],
                    verification_command="SHOW log_statement;",
                ),
                SecurityAction(
                    action="Install and configure pgaudit extension",
                    sql="CREATE EXTENSION IF NOT EXISTS pgaudit;",
                    risk_level="medium",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none",
                    rollback_notes="DROP EXTENSION pgaudit;",
                    priority="scheduled",
                    compliance_frameworks=["pci_dss", "gdpr", "soc2"],
                    verification_command="SELECT * FROM pg_extension WHERE extname = 'pgaudit';",
                ),
            ],
            confidence=signal.confidence,
            references=[
                "https://www.postgresql.org/docs/current/runtime-config-logging.html",
                "github.com/pgaudit/pgaudit",
            ],
            compliance_tags=["pci_dss", "gdpr", "soc2"],
        )

    def _recommend_connection_limit(self, signal: Signal) -> SecurityRecommendation:
        """Recommendation for connection limits."""
        max_conn = signal.data.get("max_connections", 100)
        return SecurityRecommendation(
            check_name=signal.name,
            category="network_security",
            severity=signal.severity,
            title="Review Connection Limit Configuration",
            description=f"max_connections is set to {max_conn} which may be excessive.",
            current_state=f"max_connections = {max_conn}",
            desired_state="100-200 with connection pooling",
            actions=[
                SecurityAction(
                    action="Install PgBouncer for connection pooling",
                    sql=None,
                    risk_level="medium",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="brief (switchover)",
                    rollback_notes="Disable PgBouncer and restore direct connections",
                    priority="scheduled",
                    compliance_frameworks=[],
                    verification_command="ps aux | grep pgbouncer",
                ),
                SecurityAction(
                    action="Reduce max_connections",
                    sql="ALTER SYSTEM SET max_connections = 200;",
                    risk_level="high",
                    is_online=False,
                    requires_approval=True,
                    estimated_downtime="requires restart",
                    rollback_notes="ALTER SYSTEM SET max_connections = 1000;",
                    priority="scheduled",
                    compliance_frameworks=[],
                    verification_command="SHOW max_connections;",
                ),
            ],
            confidence=signal.confidence,
            references=[
                "https://www.postgresql.org/docs/current/runtime-config-connection.html",
                "pgbouncer.org/",
            ],
            compliance_tags=[],
        )

    def _recommend_superuser_reduction(self, signal: Signal) -> SecurityRecommendation:
        """Recommendation for reducing superuser accounts."""
        superusers = signal.data.get("superusers", [])
        return SecurityRecommendation(
            check_name=signal.name,
            category="authorization",
            severity=signal.severity,
            title="Reduce Number of Superuser Accounts",
            description=f"Found {len(superusers)} superuser accounts. Minimize to reduce risk.",
            current_state=f"Superusers: {', '.join(superusers)}",
            desired_state="Only necessary superuser accounts",
            actions=[
                SecurityAction(
                    action="Review superuser privileges",
                    sql="SELECT rolname FROM pg_roles WHERE rolsuper = true;",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not applicable - read operation",
                    priority="soon",
                    compliance_frameworks=["pci_dss", "soc2"],
                    verification_command="SELECT rolname FROM pg_roles WHERE rolsuper = true AND rolname NOT LIKE 'pg_%';",
                ),
                SecurityAction(
                    action="Demote non-essential superusers",
                    sql="ALTER ROLE username WITH NOSUPERUSER;",
                    risk_level="high",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none",
                    rollback_notes="ALTER ROLE username WITH SUPERUSER;",
                    priority="scheduled",
                    compliance_frameworks=["pci_dss", "soc2"],
                    verification_command="SELECT rolname FROM pg_roles WHERE rolsuper = true;",
                ),
            ],
            confidence=signal.confidence,
            references=[
                "https://www.postgresql.org/docs/current/role-attributes.html",
            ],
            compliance_tags=["pci_dss", "soc2"],
        )

    def _recommend_rls_enable(self, signal: Signal) -> SecurityRecommendation:
        """Recommendation for enabling row-level security."""
        tables = signal.data.get("sensitive_tables_without_rls", [])
        return SecurityRecommendation(
            check_name=signal.name,
            category="data_protection",
            severity=signal.severity,
            title="Enable Row-Level Security on Sensitive Tables",
            description=f"RLS not enabled on sensitive tables: {', '.join(tables)}",
            current_state="RLS disabled on sensitive tables",
            desired_state="RLS enabled with appropriate policies",
            actions=[
                SecurityAction(
                    action="Enable RLS on sensitive table",
                    sql=f"ALTER TABLE {tables[0]} ENABLE ROW LEVEL SECURITY;",
                    risk_level="medium",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none",
                    rollback_notes=f"ALTER TABLE {tables[0]} DISABLE ROW LEVEL SECURITY;",
                    priority="scheduled",
                    compliance_frameworks=["gdpr"],
                    verification_command="SELECT relname, relrowsecurity FROM pg_class WHERE relname IN ({','.join(['$' + str(i) for i in range(1, len(tables)+1)])});",
                ),
                SecurityAction(
                    action="Create RLS policy for table",
                    sql=f"CREATE POLICY select_policy ON {tables[0]} FOR SELECT USING (true);",
                    risk_level="medium",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none",
                    rollback_notes=f"DROP POLICY IF EXISTS select_policy ON {tables[0]};",
                    priority="scheduled",
                    compliance_frameworks=["gdpr"],
                    verification_command="SELECT * FROM pg_policies WHERE tablename = '{tables[0]}';",
                ),
            ],
            confidence=signal.confidence,
            references=[
                "https://www.postgresql.org/docs/current/ddl-rls.html",
            ],
            compliance_tags=["gdpr"],
        )

    def _recommend_failed_logins(self, signal: Signal) -> SecurityRecommendation:
        """Recommendation for failed login attempts."""
        count = signal.data.get("failed_attempts", 0)
        return SecurityRecommendation(
            check_name=signal.name,
            category="authentication",
            severity=signal.severity,
            title="Investigate Multiple Failed Authentication Attempts",
            description=f"Detected {count} failed authentication attempts - possible brute force attack.",
            current_state=f"{count} failed attempts logged",
            desired_state="Failed attempts within normal range with monitoring",
            actions=[
                SecurityAction(
                    action="Review failed authentication logs",
                    sql="SELECT * FROM pg_logical_extension() WHERE logfile IS NOT NULL;",  # Placeholder
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not applicable",
                    priority="immediate",
                    compliance_frameworks=["pci_dss", "soc2"],
                    verification_command="SELECT count(*) FROM pg_stat_activity WHERE state = 'failed' OR wait_event ILIKE '%client%';",
                ),
                SecurityAction(
                    action="Implement fail2ban or similar",
                    sql=None,
                    risk_level="low",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none",
                    rollback_notes="Disable fail2ban",
                    priority="soon",
                    compliance_frameworks=["pci_dss", "soc2"],
                    verification_command="systemctl status fail2ban",
                ),
                SecurityAction(
                    action="Set connection attempt limits",
                    sql="ALTER SYSTEM SET connection_limits = 5;",  # Note: this is per user
                    risk_level="medium",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none",
                    rollback_notes="ALTER SYSTEM RESET connection_limits;",
                    priority="soon",
                    compliance_frameworks=["pci_dss", "soc2"],
                    verification_command="SELECT rolname, rolconnlimit FROM pg_roles WHERE rolconnlimit > 0;",
                ),
            ],
            confidence=signal.confidence,
            references=[
                "https://www.postgresql.org/docs/current/runtime-config-connection.html",
            ],
            compliance_tags=["pci_dss", "soc2"],
        )

    def _recommend_connection_review(self, signal: Signal) -> SecurityRecommendation:
        """Recommendation for unknown host connections."""
        return SecurityRecommendation(
            check_name=signal.name,
            category="network_security",
            severity=signal.severity,
            title="Review Connections from Unknown Hosts",
            description="Database accepted connections from unrecognized hosts.",
            current_state="Connections from unknown hosts detected",
            desired_state="Only expected hosts can connect",
            actions=[
                SecurityAction(
                    action="Review pg_hba.conf host entries",
                    sql="SELECT * FROM pg_hba_file_rules() WHERE host IS NOT NULL;",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not applicable",
                    priority="soon",
                    compliance_frameworks=["pci_dss"],
                    verification_command="SELECT * FROM pg_hba_file_rules() WHERE hostaddr IS NOT NULL;",
                ),
                SecurityAction(
                    action="Update firewall rules",
                    sql=None,
                    risk_level="medium",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none",
                    rollback_notes="Restore firewall rules",
                    priority="soon",
                    compliance_frameworks=["pci_dss"],
                    verification_command="iptables -L -n | grep -E '5432|5433'",
                ),
            ],
            confidence=signal.confidence,
            references=[],
            compliance_tags=["pci_dss"],
        )

    def _recommend_sql_injection(self, signal: Signal) -> SecurityRecommendation:
        """Recommendation for SQL injection indicators."""
        pattern = signal.data.get("pattern", "")
        return SecurityRecommendation(
            check_name=signal.name,
            category="access_control",
            severity=signal.severity,
            title="Investigate Potential SQL Injection Attempts",
            description=f"Detected potential SQL injection pattern: {pattern}",
            current_state="Suspicious query patterns detected",
            desired_state="Application uses parameterized queries",
            actions=[
                SecurityAction(
                    action="Review application logs for injection attempts",
                    sql=None,
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not applicable",
                    priority="immediate",
                    compliance_frameworks=["pci_dss"],
                    verification_command="SELECT * FROM pg_log WHERE message ILIKE '%OR 1=1%' OR message ILIKE '%UNION SELECT%';",
                ),
                SecurityAction(
                    action="Audit application code for SQL injection vulnerabilities",
                    sql=None,
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not applicable",
                    priority="immediate",
                    compliance_frameworks=["pci_dss"],
                    verification_command=None,
                ),
                SecurityAction(
                    action="Enable query logging for suspicious patterns",
                    sql="ALTER SYSTEM SET log_min_error_statement = 'notice'; SELECT pg_reload_conf();",
                    risk_level="low",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none",
                    rollback_notes="ALTER SYSTEM SET log_min_error_statement = 'error';",
                    priority="soon",
                    compliance_frameworks=["pci_dss", "soc2"],
                    verification_command="SHOW log_min_error_statement;",
                ),
            ],
            confidence=signal.confidence,
            references=[
                "https://www.postgresql.org/docs/current/sql-prepare.html",
                "owasp.org/www-community/attacks/SQL_Injection",
            ],
            compliance_tags=["pci_dss"],
        )

    def _recommend_ssl_review(self, signal: Signal) -> SecurityRecommendation:
        """Recommendation for SSL connection errors."""
        return SecurityRecommendation(
            check_name=signal.name,
            category="encryption",
            severity=signal.severity,
            title="Review SSL Connection Errors",
            description="SSL connection errors detected - may indicate certificate or configuration issues.",
            current_state="SSL errors in logs",
            desired_state="Healthy SSL connections",
            actions=[
                SecurityAction(
                    action="Check SSL certificate expiration",
                    sql="SELECT *, pg_stat_ssl.cert_not_after - now() as days_until_expiry FROM pg_stat_ssl JOIN pg_stat_activity ON pg_stat_ssl.pid = pg_stat_activity.pid WHERE ssl = true;",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not applicable",
                    priority="soon",
                    compliance_frameworks=["pci_dss", "gdpr"],
                    verification_command="SELECT ssl, count(*) FROM pg_stat_ssl GROUP BY ssl;",
                ),
                SecurityAction(
                    action="Renew SSL certificate if expired",
                    sql=None,  # External process
                    risk_level="medium",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="brief (certificate reload)",
                    rollback_notes="Restore previous certificate",
                    priority="immediate",
                    compliance_frameworks=["pci_dss", "gdpr"],
                    verification_command="SELECT pg_stat_ssl.cert_not_after FROM pg_stat_ssl WHERE ssl = true LIMIT 1;",
                ),
            ],
            confidence=signal.confidence,
            references=[
                "https://www.postgresql.org/docs/current/ssl-tcp.html",
            ],
            compliance_tags=["pci_dss", "gdpr"],
        )

    def _recommend_audit_role_changes(self, signal: Signal) -> SecurityRecommendation:
        """Recommendation for role modification auditing."""
        return SecurityRecommendation(
            check_name=signal.name,
            category="authorization",
            severity=signal.severity,
            title="Audit Role Modification Changes",
            description="Role changes detected - ensure these are authorized and documented.",
            current_state="Role modifications logged",
            desired_state="Role changes with proper authorization and audit trail",
            actions=[
                SecurityAction(
                    action="Review recent role changes",
                    sql="SELECT * FROM pg_audit_log WHERE action ILIKE '%ROLE%';",
                    risk_level="low",
                    is_online=True,
                    requires_approval=False,
                    estimated_downtime="none",
                    rollback_notes="Not applicable",
                    priority="soon",
                    compliance_frameworks=["soc2"],
                    verification_command="SELECT * FROM pg_roles WHERE rolname NOT LIKE 'pg_%' AND rolcanlogin = true;",
                ),
                SecurityAction(
                    action="Implement role change alerts",
                    sql=None,
                    risk_level="low",
                    is_online=True,
                    requires_approval=True,
                    estimated_downtime="none",
                    rollback_notes="Disable alerts",
                    priority="scheduled",
                    compliance_frameworks=["soc2"],
                    verification_command="SELECT * FROM pg_event_trigger_ddl_commands() WHERE command_tag = 'DROP ROLE' OR command_tag = 'ALTER ROLE';",
                ),
            ],
            confidence=signal.confidence,
            references=[],
            compliance_tags=["soc2"],
        )

    def _build_compliance_summary(
        self,
        recommendations: List[SecurityRecommendation]
    ) -> Dict[str, Dict[str, Any]]:
        """Build compliance framework summary."""
        frameworks = ["pci_dss", "gdpr", "soc2"]
        summary = {}

        for fw in frameworks:
            fw_recs = [r for r in recommendations if fw in r.compliance_tags]
            passed = sum(1 for r in fw_recs if r.severity in ["low", "info"])
            total = len(fw_recs) or 1

            summary[fw] = {
                "score": (passed / total) * 100,
                "recommendations_count": len(fw_recs),
                "status": "compliant" if (passed / total) > 0.7 else "needs_attention",
            }

        return summary

    def _build_implementation_plan(
        self,
        recommendations: List[SecurityRecommendation]
    ) -> List[Dict[str, Any]]:
        """Build phased implementation plan."""
        plan = []

        # Phase 1: Critical and High (immediate)
        phase1 = [r for r in recommendations if r.severity in ["critical", "high"]]
        if phase1:
            plan.append({
                "phase": 1,
                "name": "Critical Security Issues",
                "priority": "immediate",
                "recommendations": [r.title for r in phase1],
                "actions": [
                    a for r in phase1 for a in r.actions
                    if a.priority in ["immediate"]
                ],
            })

        # Phase 2: Medium (soon)
        phase2 = [r for r in recommendations if r.severity == "medium"]
        if phase2:
            plan.append({
                "phase": 2,
                "name": "Medium Priority Security Improvements",
                "priority": "soon",
                "recommendations": [r.title for r in phase2],
                "actions": [
                    a for r in phase2 for a in r.actions
                    if a.priority in ["immediate", "soon"]
                ],
            })

        # Phase 3: Low (scheduled)
        phase3 = [r for r in recommendations if r.severity in ["low", "info"]]
        if phase3:
            plan.append({
                "phase": 3,
                "name": "Low Priority Security Enhancements",
                "priority": "scheduled",
                "recommendations": [r.title for r in phase3],
                "actions": [
                    a for r in phase3 for a in r.actions
                    if a.priority in ["immediate", "soon", "scheduled"]
                ],
            })

        return plan


# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    from signals.security_signals import SecuritySignalGenerator

    sample_config = {
        "ssl": "off",
        "listen_addresses": "*",
        "password_encryption": "md5",
        "log_connections": "off",
        "log_disconnections": "off",
        "log_statement": "none",
        "shared_preload_libraries": "",
        "max_connections": 100,
    }

    # Generate security report
    generator = SecuritySignalGenerator()
    signal_result = generator.process(sample_config)

    # Get recommendations
    recommender = SecurityRecommender()
    report = recommender.recommend(signal_result)

    print(f"Security Score: {report.overall_score:.1f}%")
    print(f"Critical: {report.critical_count}, High: {report.high_count}, Medium: {report.medium_count}")
    print(f"\nCompliance Summary: {report.compliance_summary}")

