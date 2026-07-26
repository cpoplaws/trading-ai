"""
Security Audit - Production security checks and best practices

Features:
- Private key security validation
- API credential management
- Environment variable security
- File permission checks
- Code pattern analysis for security issues
"""

import os
import stat
import logging
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security issue severity"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityIssue:
    """Security issue found during audit"""
    level: SecurityLevel
    category: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    recommendation: Optional[str] = None


class SecurityAuditor:
    """
    Security auditing system for trading platform.

    Checks:
    - Private key handling
    - API credential security
    - File permissions
    - Environment variable security
    - Common security anti-patterns
    """

    # Patterns that should never appear in code
    DANGEROUS_PATTERNS = [
        (r'(private[_-]?key|secret[_-]?key|api[_-]?secret)\s*=\s*["\'][^"\']{10,}["\']',
         "Hardcoded credential detected",
         SecurityLevel.CRITICAL),
        (r'password\s*=\s*["\'][^"\']{4,}["\']',
         "Hardcoded password detected",
         SecurityLevel.CRITICAL),
        (r'BEGIN (RSA |EC )?PRIVATE KEY',
         "Private key in plain text",
         SecurityLevel.CRITICAL),
        (r'eval\s*\(',
         "Use of eval() detected (code injection risk)",
         SecurityLevel.HIGH),
        (r'exec\s*\(',
         "Use of exec() detected (code injection risk)",
         SecurityLevel.HIGH),
        (r'pickle\.loads?\s*\(',
         "Use of pickle (deserialization risk)",
         SecurityLevel.MEDIUM),
    ]

    # Files that should have restricted permissions
    SENSITIVE_FILES = [
        ".env",
        "wallets.enc",
        "**/wallets.enc",
        "**/*.key",
        "**/*.pem",
    ]

    # Environment variables that should be set
    REQUIRED_ENV_VARS = [
        "WALLET_MASTER_PASSWORD",
    ]

    def __init__(self, project_root: str = "."):
        """
        Initialize security auditor.

        Args:
            project_root: Root directory of project
        """
        self.project_root = Path(project_root).resolve()
        self.issues: List[SecurityIssue] = []

    def run_full_audit(self) -> List[SecurityIssue]:
        """
        Run complete security audit.

        Returns:
            List of security issues found
        """
        self.issues = []

        logger.info("Starting security audit...")

        # Check 1: Environment variables
        self._check_environment_variables()

        # Check 2: File permissions
        self._check_file_permissions()

        # Check 3: Code patterns
        self._scan_code_patterns()

        # Check 4: Gitignore
        self._check_gitignore()

        # Check 5: Dependencies
        self._check_dependencies()

        # Summary
        self._print_summary()

        return self.issues

    def _check_environment_variables(self) -> None:
        """Check environment variable security."""
        logger.info("Checking environment variables...")

        # Check required variables
        for var in self.REQUIRED_ENV_VARS:
            if not os.getenv(var):
                self.issues.append(SecurityIssue(
                    level=SecurityLevel.HIGH,
                    category="Environment Variables",
                    description=f"Required environment variable not set: {var}",
                    recommendation=f"Set {var} in your .env file or environment"
                ))

        # Check for exposed credentials
        env_file = self.project_root / ".env"
        if env_file.exists():
            # Check .env permissions
            st = os.stat(env_file)
            mode = st.st_mode

            # Check if world-readable
            if mode & stat.S_IROTH:
                self.issues.append(SecurityIssue(
                    level=SecurityLevel.HIGH,
                    category="File Permissions",
                    description=".env file is world-readable",
                    file_path=str(env_file),
                    recommendation="Run: chmod 600 .env"
                ))

        else:
            self.issues.append(SecurityIssue(
                level=SecurityLevel.INFO,
                category="Environment Variables",
                description=".env file not found",
                recommendation="Create .env file for local configuration"
            ))

    def _check_file_permissions(self) -> None:
        """Check sensitive file permissions."""
        logger.info("Checking file permissions...")

        for pattern in self.SENSITIVE_FILES:
            for file_path in self.project_root.rglob(pattern.lstrip("**/")):
                if not file_path.is_file():
                    continue

                st = os.stat(file_path)
                mode = st.st_mode

                # Check if group or world readable/writable
                if mode & (stat.S_IRWXG | stat.S_IRWXO):
                    self.issues.append(SecurityIssue(
                        level=SecurityLevel.HIGH,
                        category="File Permissions",
                        description=f"Sensitive file has insecure permissions: {file_path.name}",
                        file_path=str(file_path),
                        recommendation=f"Run: chmod 600 {file_path}"
                    ))

    def _scan_code_patterns(self) -> None:
        """Scan code for dangerous patterns."""
        logger.info("Scanning code patterns...")

        # Scan Python files
        python_files = list(self.project_root.rglob("*.py"))

        for file_path in python_files:
            # Skip virtual environments and test files
            if any(skip in str(file_path) for skip in ["venv", ".venv", "site-packages"]):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                    for pattern, description, level in self.DANGEROUS_PATTERNS:
                        for line_num, line in enumerate(lines, 1):
                            if re.search(pattern, line, re.IGNORECASE):
                                # Check if it's in a comment
                                if line.strip().startswith('#'):
                                    continue

                                self.issues.append(SecurityIssue(
                                    level=level,
                                    category="Code Security",
                                    description=description,
                                    file_path=str(file_path.relative_to(self.project_root)),
                                    line_number=line_num,
                                    recommendation="Use environment variables or secure key management"
                                ))

            except Exception as e:
                logger.debug(f"Error scanning {file_path}: {e}")

    def _check_gitignore(self) -> None:
        """Check .gitignore for sensitive files."""
        logger.info("Checking .gitignore...")

        gitignore = self.project_root / ".gitignore"

        # Required entries
        required_entries = [
            ".env",
            "*.key",
            "*.pem",
            "**/wallets.enc",
            "**/*.enc",
        ]

        if not gitignore.exists():
            self.issues.append(SecurityIssue(
                level=SecurityLevel.MEDIUM,
                category="Git Security",
                description=".gitignore file not found",
                recommendation="Create .gitignore to prevent committing sensitive files"
            ))
            return

        try:
            with open(gitignore, 'r') as f:
                content = f.read()

            for entry in required_entries:
                if entry not in content:
                    self.issues.append(SecurityIssue(
                        level=SecurityLevel.MEDIUM,
                        category="Git Security",
                        description=f"Sensitive pattern not in .gitignore: {entry}",
                        file_path=str(gitignore),
                        recommendation=f"Add '{entry}' to .gitignore"
                    ))

        except Exception as e:
            logger.error(f"Error reading .gitignore: {e}")

    def _check_dependencies(self) -> None:
        """Check dependency security."""
        logger.info("Checking dependencies...")

        requirements_file = self.project_root / "requirements.txt"

        if not requirements_file.exists():
            self.issues.append(SecurityIssue(
                level=SecurityLevel.LOW,
                category="Dependencies",
                description="requirements.txt not found",
                recommendation="Create requirements.txt to pin dependency versions"
            ))
            return

        try:
            with open(requirements_file, 'r') as f:
                content = f.read()

            # Check for unpinned versions
            lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]

            for line in lines:
                if '==' not in line:
                    self.issues.append(SecurityIssue(
                        level=SecurityLevel.LOW,
                        category="Dependencies",
                        description=f"Unpinned dependency: {line}",
                        file_path="requirements.txt",
                        recommendation="Pin all dependency versions (e.g., package==1.2.3)"
                    ))

        except Exception as e:
            logger.error(f"Error reading requirements.txt: {e}")

    def _print_summary(self) -> None:
        """Print audit summary."""
        print("\n" + "="*70)
        print("SECURITY AUDIT SUMMARY")
        print("="*70)

        # Count by severity
        severity_counts = {level: 0 for level in SecurityLevel}
        for issue in self.issues:
            severity_counts[issue.level] += 1

        print(f"\nTotal issues found: {len(self.issues)}")
        print("\nBy severity:")
        for level in SecurityLevel:
            count = severity_counts[level]
            if count > 0:
                emoji = {
                    SecurityLevel.INFO: "ℹ️",
                    SecurityLevel.LOW: "🟡",
                    SecurityLevel.MEDIUM: "🟠",
                    SecurityLevel.HIGH: "🔴",
                    SecurityLevel.CRITICAL: "🚨"
                }
                print(f"  {emoji[level]} {level.value.upper()}: {count}")

        # Print issues
        if self.issues:
            print("\nIssues:")
            for i, issue in enumerate(self.issues, 1):
                print(f"\n{i}. [{issue.level.value.upper()}] {issue.category}")
                print(f"   {issue.description}")
                if issue.file_path:
                    location = f"{issue.file_path}"
                    if issue.line_number:
                        location += f":{issue.line_number}"
                    print(f"   Location: {location}")
                if issue.recommendation:
                    print(f"   ✓ {issue.recommendation}")

        else:
            print("\n✅ No security issues found!")

        print("\n" + "="*70)

    def get_critical_issues(self) -> List[SecurityIssue]:
        """Get critical and high severity issues."""
        return [
            issue for issue in self.issues
            if issue.level in [SecurityLevel.CRITICAL, SecurityLevel.HIGH]
        ]

    def export_report(self, output_file: str = "security_audit_report.txt") -> None:
        """Export audit report to file."""
        with open(output_file, 'w') as f:
            f.write("SECURITY AUDIT REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Date: {__import__('datetime').datetime.now()}\n")
            f.write(f"Project: {self.project_root}\n\n")

            f.write(f"Total issues: {len(self.issues)}\n\n")

            for issue in self.issues:
                f.write(f"[{issue.level.value.upper()}] {issue.category}\n")
                f.write(f"Description: {issue.description}\n")
                if issue.file_path:
                    f.write(f"File: {issue.file_path}")
                    if issue.line_number:
                        f.write(f":{issue.line_number}")
                    f.write("\n")
                if issue.recommendation:
                    f.write(f"Recommendation: {issue.recommendation}\n")
                f.write("\n")

        logger.info(f"Security audit report exported to {output_file}")


def validate_wallet_security(wallet_file: str, master_password: str) -> List[SecurityIssue]:
    """
    Validate wallet security setup.

    Args:
        wallet_file: Path to wallet file
        master_password: Master password

    Returns:
        List of security issues
    """
    issues = []

    # Check password strength
    if len(master_password) < 12:
        issues.append(SecurityIssue(
            level=SecurityLevel.HIGH,
            category="Wallet Security",
            description="Master password too short (minimum 12 characters)",
            recommendation="Use a password with at least 12 characters"
        ))

    # Check for common patterns
    common_passwords = ["password", "123456", "test", "admin"]
    if any(pwd in master_password.lower() for pwd in common_passwords):
        issues.append(SecurityIssue(
            level=SecurityLevel.CRITICAL,
            category="Wallet Security",
            description="Master password uses common pattern",
            recommendation="Use a strong, unique password"
        ))

    # Check file permissions if file exists
    if os.path.exists(wallet_file):
        st = os.stat(wallet_file)
        mode = st.st_mode

        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            issues.append(SecurityIssue(
                level=SecurityLevel.CRITICAL,
                category="Wallet Security",
                description=f"Wallet file has insecure permissions",
                file_path=wallet_file,
                recommendation=f"Run: chmod 600 {wallet_file}"
            ))

    return issues


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*70)
    print("SECURITY AUDIT TEST")
    print("="*70)

    # Run full audit
    auditor = SecurityAuditor(project_root="/Users/silasmarkowicz/trading-ai-working")
    issues = auditor.run_full_audit()

    # Check critical issues
    critical = auditor.get_critical_issues()
    if critical:
        print(f"\n⚠️  {len(critical)} critical/high severity issues require immediate attention!")

    # Export report
    auditor.export_report()
    print("\n✅ Security audit complete!")
