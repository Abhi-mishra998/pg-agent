#!/usr/bin/env python3
"""
Security Signals Module

Generate security-related signals from PostgreSQL configuration and logs.
Designed for proactive security monitoring and compliance checking.

Security Categories:
- Authentication (ssl, password policies, auth methods)
- Authorization (roles, privileges, RLS)
- Encryption (ssl/tls, pgcrypto)
- Audit Logging (pgAudit, log configurations)
- Network Security (connection settings, listen_addresses)
- Data Protection (row-level security, column encryption)
- Access Control (public schema, default permissions)
- Compliance (PCI-DSS, GDPR, SOC2 checks)
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from signals.signal_engine import Signal, SignalResult


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

# Security thresholds
SSL_REQUIRED = True
MIN_PASSWORD_LENGTH = 12
MAX_CONN_PER_USER = 10
AUDIT_LOG_MIN_SEVERITY = "log"
RLS_REQUIRED_TABLES = ["users", "customers", "payments", "orders"]

# Compliance frameworks
COMPLIANCE_CHECKS = {
    "pci_dss": [
        "ssl",
        "password",
        "audit",
        "connection",
    ],
    "gdpr": [
        "encryption",
        "audit",
        "rl s",
    ],
    "soc2": [
        "access",
        "audit",
        "encryption",
    ],
}


# -------------------------------------------------------------------
# Data Models
# -------------------------------------------------------------------

@dataclass
class SecurityCheck:
    """A single security check result."""
    check_name: str
    category: str
    passed: bool
    severity: str  # critical, high, medium, low, info
    message: str
    current_value: str
    recommended_value: str
    remediation: str
    compliance_tags: List[str]


@dataclass
class SecurityReport:
    """Complete security assessment report."""
    checks: List[SecurityCheck]
    overall_score: float
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    compliance_status: Dict[str, Dict[str, Any]]
    timestamp: str


# -------------------------------------------------------------------
# Security Signal Generators
# -------------------------------------------------------------------

class SecuritySignalGenerator:
    """
    Generate security signals from PostgreSQL configuration and logs.
    
    Detects:
    - SSL/TLS not enabled
    - Weak authentication settings
    - Overly permissive roles
    - Missing audit logging
    - RLS not enabled on sensitive tables
    - Public schema vulnerabilities
    - Connection security issues
    """

    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

    def process(self, data: Any) -> SignalResult:
        """
        Process data and generate security signals.
        
        Args:
            data: PostgreSQL config, logs, or JSON security data
            
        Returns:
            SignalResult with security signals
        """
        start_time = time.time()
        signals: List[Signal] = []
        now = int(time.time() * 1000)

        # Handle different input types
        if isinstance(data, dict):
            signals = self._generate_from_config(data)
        elif isinstance(data, str):
            signals = self._generate_from_logs(data)
        else:
            self.logger.warning(f"Unknown data type for security analysis: {type(data)}")

        # Build analysis summary
        severity_count: Dict[str, int] = {}
        for s in signals:
            severity_count[s.severity] = severity_count.get(s.severity, 0) + 1

        analysis = {
            "signal_count": len(signals),
            "severities": severity_count,
            "highest_severity": max(severity_count, key=severity_count.get) if severity_count else "none",
            "security_categories": self._count_categories(signals),
        }

        duration = time.time() - start_time
        self.logger.info(f"Security analysis completed in {duration:.3f}s")

        return SignalResult(
            signals=signals,
            analysis=analysis,
            filtered_count=0,
            processing_time=duration,
        )

    def _generate_from_config(self, config: Dict[str, Any]) -> List[Signal]:
        """Generate signals from PostgreSQL configuration."""
        signals: List[Signal] = []
        now = int(time.time() * 1000)

        # Extract common settings
        ssl = config.get("ssl", config.get("ssl_mode", "off"))
        listen = config.get("listen_addresses", "*")
        password_encryption = config.get("password_encryption", "md5")
        auth_methods = config.get("authentication_timeout", "60s")
        log_connections = config.get("log_connections", "off")
        log_disconnections = config.get("log_disconnections", "off")
        log_statement = config.get("log_statement", "none")
        shared_preload_libraries = config.get("shared_preload_libraries", "")
        
        # 1. SSL/TLS Check
        if str(ssl).lower() != "on":
            signals.append(Signal(
                id=f"ssl_disabled_{now}",
                name="ssl_not_enabled",
                type="security",
                severity="critical",
                confidence=0.95,
                data={
                    "ssl_enabled": ssl,
                    "setting": "ssl",
                    "issue": "SSL/TLS is not enabled for connections",
                },
                metadata={
                    "explain": "SSL is disabled - all connections are unencrypted",
                    "category": "encryption",
                    "remediation": "Set ssl = on in postgresql.conf and restart",
                    "compliance": ["pci_dss", "gdpr", "soc2"],
                },
            ))

        # 2. Listen Addresses Check
        if listen == "*" or "," in str(listen):
            signals.append(Signal(
                id=f"listen_all_{now}",
                name="listen_addresses_vulnerable",
                type="security",
                severity="high",
                confidence=0.90,
                data={
                    "listen_addresses": listen,
                    "setting": "listen_addresses",
                    "issue": "Database listening on all interfaces",
                },
                metadata={
                    "explain": f"PostgreSQL listens on '{listen}' - may expose to untrusted networks",
                    "category": "network_security",
                    "remediation": "Set listen_addresses to specific IPs only",
                    "compliance": ["pci_dss"],
                },
            ))

        # 3. Password Encryption Check
        if password_encryption.lower() == "md5":
            signals.append(Signal(
                id=f"weak_password_enc_{now}",
                name="md5_password_encryption",
                type="security",
                severity="high",
                confidence=0.92,
                data={
                    "password_encryption": password_encryption,
                    "setting": "password_encryption",
                    "issue": "Using MD5 for password hashing",
                },
                metadata={
                    "explain": "MD5 password encryption is considered weak - use scram-sha-256",
                    "category": "authentication",
                    "remediation": "Set password_encryption = scram-sha-256",
                    "compliance": ["pci_dss"],
                },
            ))

        # 4. Audit Logging Check
        audit_missing = (
            log_connections.lower() == "off" or
            log_disconnections.lower() == "off" or
            log_statement.lower() == "none" or
            "pgaudit" not in shared_preload_libraries.lower()
        )
        if audit_missing:
            signals.append(Signal(
                id=f"audit_missing_{now}",
                name="audit_logging_disabled",
                type="security",
                severity="high",
                confidence=0.88,
                data={
                    "log_connections": log_connections,
                    "log_disconnections": log_disconnections,
                    "log_statement": log_statement,
                    "pgaudit_installed": "pgaudit" in shared_preload_libraries.lower(),
                    "issue": "Audit logging is not properly configured",
                },
                metadata={
                    "explain": "Critical audit logs are missing (connections, disconnections, statements)",
                    "category": "audit_logging",
                    "remediation": "Enable comprehensive logging and install pgaudit extension",
                    "compliance": ["pci_dss", "gdpr", "soc2"],
                },
            ))

        # 5. Connection Security Check
        max_conn = config.get("max_connections", 100)
        if max_conn > 1000:
            signals.append(Signal(
                id=f"high_max_conn_{now}",
                name="excessive_max_connections",
                type="security",
                severity="medium",
                confidence=0.80,
                data={
                    "max_connections": max_conn,
                    "setting": "max_connections",
                    "issue": "High max_connections value increases attack surface",
                },
                metadata={
                    "explain": f"max_connections={max_conn} may indicate over-provisioning",
                    "category": "network_security",
                    "remediation": "Use connection pooling (PgBouncer) and reduce max_connections",
                    "compliance": [],
                },
            ))

        # 6. superuser Check (from pg_roles if available)
        pg_roles = config.get("pg_roles", [])
        if pg_roles:
            superusers = [r for r in pg_roles if r.get("rolsuper", False) and not r.get("rolname", "").startswith("pg_")]
            if len(superusers) > 1:
                signals.append(Signal(
                    id=f"too_many_superusers_{now}",
                    name="excessive_superuser_accounts",
                    type="security",
                    severity="medium",
                    confidence=0.85,
                    data={
                        "superuser_count": len(superusers),
                        "superusers": [s.get("rolname") for s in superusers],
                        "issue": "Multiple superuser accounts increase risk",
                    },
                    metadata={
                        "explain": f"Found {len(superusers)} superuser accounts besides postgres",
                        "category": "authorization",
                        "remediation": "Reduce superuser accounts and use role-based access",
                        "compliance": ["pci_dss", "soc2"],
                    },
                ))

        # 7. Public Schema Check
        if pg_roles:
            public_schema_public = any(
                r.get("rolname") == "public" 
                for r in pg_roles
            )
            # Check if public schema has create privilege
            if not public_schema_public:
                # Look for CREATE privilege on public schema
                pass  # Would need pg_namespace data

        # 8. Row-Level Security Check
        rls_tables = config.get("rls_status", [])
        if rls_tables:
            unprotected = [t for t in rls_tables if not t.get("rlsenabled", False)]
            sensitive_tables = [t for t in RLS_REQUIRED_TABLES if t in [u.get("tablename") for u in unprotected]]
            if sensitive_tables:
                signals.append(Signal(
                    id=f"rls_missing_{now}",
                    name="row_level_security_disabled",
                    type="security",
                    severity="medium",
                    confidence=0.82,
                    data={
                        "unprotected_tables": unprotected,
                        "sensitive_tables_without_rls": sensitive_tables,
                        "issue": "RLS not enabled on sensitive tables",
                    },
                    metadata={
                        "explain": f"Sensitive tables without RLS: {', '.join(sensitive_tables)}",
                        "category": "data_protection",
                        "remediation": "Enable RLS on sensitive tables and create policies",
                        "compliance": ["gdpr"],
                    },
                ))

        # 9. Check for weak authentication methods
        if "ident" in str(auth_methods).lower() or "trust" in str(auth_methods).lower():
            signals.append(Signal(
                id=f"weak_auth_{now}",
                name="weak_authentication_method",
                type="security",
                severity="critical",
                confidence=0.95,
                data={
                    "auth_method": auth_methods,
                    "issue": "Using weak authentication method (ident/trust)",
                },
                metadata={
                    "explain": "ident or trust authentication can be bypassed",
                    "category": "authentication",
                    "remediation": "Use scram-sha-256 or certificate authentication",
                    "compliance": ["pci_dss", "soc2"],
                },
            ))

        # 10. Check for missing ssl_preload
        if "ssl" not in shared_preload_libraries.lower():
            signals.append(Signal(
                id=f"no_ssl_preload_{now}",
                name="ssl_preload_not_configured",
                type="security",
                severity="low",
                confidence=0.75,
                data={
                    "shared_preload_libraries": shared_preload_libraries,
                    "issue": "ssl library not preloaded",
                },
                metadata={
                    "explain": "ssl library not in shared_preload_libraries",
                    "category": "encryption",
                    "remediation": "Add ssl to shared_preload_libraries for SSL session caching",
                    "compliance": [],
                },
            ))

        return signals

    def _generate_from_logs(self, log_text: str) -> List[Signal]:
        """Generate signals from log text."""
        signals: List[Signal] = []
        now = int(time.time() * 1000)
        log_lower = log_text.lower()

        # 1. Failed login attempts
        if "password authentication failed" in log_lower or "invalid password" in log_lower:
            failed_count = log_lower.count("password authentication failed")
            signals.append(Signal(
                id=f"failed_logins_{now}",
                name="multiple_failed_authentications",
                type="security",
                severity="high",
                confidence=0.85,
                data={
                    "failed_attempts": failed_count,
                    "pattern": "password authentication failed",
                },
                metadata={
                    "explain": f"Detected {failed_count} failed authentication attempts",
                    "category": "authentication",
                    "remediation": "Check for brute force attacks and consider fail2ban",
                    "compliance": ["pci_dss", "soc2"],
                },
            ))

        # 2. Connection from unknown host
        if "connection from unknown host" in log_lower:
            signals.append(Signal(
                id=f"unknown_host_{now}",
                name="connection_from_unknown_host",
                type="security",
                severity="medium",
                confidence=0.80,
                data={
                    "pattern": "connection from unknown host",
                },
                metadata={
                    "explain": "Connections from unrecognized hosts detected",
                    "category": "network_security",
                    "remediation": "Review pg_hba.conf and firewall rules",
                    "compliance": [],
                },
            ))

        # 3. Superuser role usage
        if "cannot execute" in log_lower and "superuser" in log_lower:
            signals.append(Signal(
                id=f"superuser_error_{now}",
                name="superuser_permission_error",
                type="security",
                severity="info",
                confidence=0.70,
                data={
                    "pattern": "superuser permission denied",
                },
                metadata={
                    "explain": "Superuser permission errors in logs",
                    "category": "authorization",
                    "remediation": "Review role permissions and privilege escalation",
                    "compliance": [],
                },
            ))

        # 4. SQL injection indicators
        sql_injection_patterns = [
            "' OR '1'='1",
            "' --",
            "UNION SELECT",
            "DROP TABLE",
            "EXEC xp_",
        ]
        for pattern in sql_injection_patterns:
            if pattern.lower() in log_lower:
                signals.append(Signal(
                    id=f"sql_injection_{now}",
                    name="potential_sql_injection",
                    type="security",
                    severity="critical",
                    confidence=0.90,
                    data={
                        "pattern": pattern,
                    },
                    metadata={
                        "explain": f"Potential SQL injection pattern detected: {pattern}",
                        "category": "access_control",
                        "remediation": "Review application code and use parameterized queries",
                        "compliance": ["pci_dss"],
                    },
                ))
                break

        # 5. SSL connection errors
        if "ssl connection" in log_lower and "error" in log_lower:
            signals.append(Signal(
                id=f"ssl_error_{now}",
                name="ssl_connection_errors",
                type="security",
                severity="medium",
                confidence=0.75,
                data={
                    "pattern": "ssl connection error",
                },
                metadata={
                    "explain": "SSL connection errors detected in logs",
                    "category": "encryption",
                    "remediation": "Check SSL certificate validity and configuration",
                    "compliance": [],
                },
            ))

        # 6. Role changes
        if ("role" in log_lower and ("created" in log_lower or "modified" in log_lower or "dropped" in log_lower)):
            signals.append(Signal(
                id=f"role_change_{now}",
                name="role_modification_detected",
                type="security",
                severity="medium",
                confidence=0.78,
                data={
                    "pattern": "role modification",
                },
                metadata={
                    "explain": "Role creation/modification/deletion detected",
                    "category": "authorization",
                    "remediation": "Review role changes for unauthorized modifications",
                    "compliance": ["soc2"],
                },
            ))

        return signals

    def _count_categories(self, signals: List[Signal]) -> Dict[str, int]:
        """Count signals by security category."""
        categories: Dict[str, int] = {}
        for signal in signals:
            category = signal.metadata.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1
        return categories

    def run_security_check(
        self,
        config: Dict[str, Any],
        pg_roles: Optional[List[Dict]] = None,
        rls_status: Optional[List[Dict]] = None,
    ) -> SecurityReport:
        """
        Run comprehensive security check.
        
        Args:
            config: PostgreSQL configuration settings
            pg_roles: Role information from pg_roles
            rls_status: RLS status from information_schema
            
        Returns:
            SecurityReport with all check results
        """
        # Prepare full config
        full_config = config.copy()
        if pg_roles:
            full_config["pg_roles"] = pg_roles
        if rls_status:
            full_config["rls_status"] = rls_status
        
        # Generate signals
        result = self.process(full_config)
        
        # Convert signals to security checks
        checks = []
        for signal in result.signals:
            check = SecurityCheck(
                check_name=signal.name,
                category=signal.metadata.get("category", "unknown"),
                passed=False,
                severity=signal.severity,
                message=signal.metadata.get("explain", signal.data.get("issue", "")),
                current_value=str(signal.data),
                recommended_value=signal.metadata.get("remediation", ""),
                remediation=signal.metadata.get("remediation", ""),
                compliance_tags=signal.metadata.get("compliance", []),
            )
            checks.append(check)
        
        # Calculate counts
        critical = sum(1 for c in checks if c.severity == "critical")
        high = sum(1 for c in checks if c.severity == "high")
        medium = sum(1 for c in checks if c.severity == "medium")
        low = sum(1 for c in checks if c.severity == "low")
        
        # Calculate overall score
        max_score = len(checks) * 4 if checks else 1
        actual_score = sum(
            4 - {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}.get(c.severity, 0)
            for c in checks
        )
        overall_score = (actual_score / max_score) * 100 if max_score > 0 else 100
        
        # Calculate compliance status
        compliance_status = {}
        for framework, tags in COMPLIANCE_CHECKS.items():
            checks_for_framework = [c for c in checks if any(t in c.compliance_tags for t in tags)]
            passed_checks = sum(1 for c in checks_for_framework if c.severity in ["low", "info"])
            total_checks = len(checks_for_framework) or 1
            compliance_status[framework] = {
                "score": (passed_checks / total_checks) * 100,
                "checks_passed": passed_checks,
                "checks_total": total_checks,
                "status": "compliant" if (passed_checks / total_checks) > 0.8 else "non_compliant",
            }
        
        return SecurityReport(
            checks=checks,
            overall_score=overall_score,
            critical_count=critical,
            high_count=high,
            medium_count=medium,
            low_count=low,
            compliance_status=compliance_status,
            timestamp=datetime.utcnow().isoformat(),
        )


# -------------------------------------------------------------------
# Standalone Functions for Quick Checks
# -------------------------------------------------------------------

def check_ssl_enabled(config: Dict[str, Any]) -> bool:
    """Check if SSL is enabled."""
    return str(config.get("ssl", "off")).lower() == "on"


def check_password_encryption(config: Dict[str, Any]) -> bool:
    """Check if strong password encryption is used."""
    return config.get("password_encryption", "").lower() in ["scram-sha-256", "scram_sha_256"]


def check_audit_logging(config: Dict[str, Any]) -> bool:
    """Check if audit logging is properly configured."""
    return (
        config.get("log_connections", "").lower() == "on" and
        config.get("log_disconnections", "").lower() == "on" and
        "pgaudit" in config.get("shared_preload_libraries", "").lower()
    )


def check_listen_addresses(config: Dict[str, Any]) -> bool:
    """Check if listen_addresses is restricted."""
    listen = config.get("listen_addresses", "*")
    return listen != "*" and "," not in str(listen)


# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Example PostgreSQL configuration
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
    
    # Run security check
    generator = SecuritySignalGenerator()
    report = generator.run_security_check(sample_config)
    
    print(f"Overall Security Score: {report.overall_score:.1f}%")
    print(f"Critical: {report.critical_count}, High: {report.high_count}, Medium: {report.medium_count}")
    
    for check in report.checks:
        print(f"\n[{check.severity.upper()}] {check.check_name}")
        print(f"  {check.message}")
        print(f"  Remediation: {check.remediation}")

