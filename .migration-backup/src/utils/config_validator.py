"""
Configuration validator and security scanner for trading-ai.
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


class ConfigValidator:
    """Validates configuration files and checks for security issues."""

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.issues: List[Dict[str, str]] = []

    def validate_env_file(self) -> Tuple[bool, List[str]]:
        """Validate .env file exists and has correct format."""
        errors = []
        env_template = self.root_path / ".env.template"
        env_file = self.root_path / ".env"

        if not env_template.exists():
            errors.append("❌ .env.template not found")
            return False, errors

        if not env_file.exists():
            errors.append("⚠️  .env file not found (will use defaults)")

        # Check for required environment variables in template
        with open(env_template) as f:
            content = f.read()
            required_vars = re.findall(r"^([A-Z_]+)=", content, re.MULTILINE)

        if not required_vars:
            errors.append("⚠️  No environment variables defined in .env.template")

        return len(errors) == 0, errors

    def validate_settings_yaml(self) -> Tuple[bool, List[str]]:
        """Validate settings.yaml file."""
        errors = []
        settings_file = self.root_path / "config" / "settings.yaml"

        if not settings_file.exists():
            errors.append("❌ config/settings.yaml not found")
            return False, errors

        try:
            with open(settings_file) as f:
                settings = yaml.safe_load(f)

            # Check for required sections
            required_sections = ["tickers", "model", "features"]
            for section in required_sections:
                if section not in settings:
                    errors.append(f"⚠️  Missing '{section}' section in settings.yaml")

        except yaml.YAMLError as e:
            errors.append(f"❌ Invalid YAML in settings.yaml: {e}")
            return False, errors

        return len(errors) == 0, errors

    def scan_for_secrets(self) -> List[Dict[str, str]]:
        """Scan for potential secrets in code."""
        secrets = []
        patterns = {
            "api_key": re.compile(r'api[_-]?key["\']?\s*[:=]\s*["\']([^"\']{20,})["\']', re.IGNORECASE),
            "password": re.compile(r'password["\']?\s*[:=]\s*["\']([^"\']{8,})["\']', re.IGNORECASE),
            "secret": re.compile(r'secret["\']?\s*[:=]\s*["\']([^"\']{20,})["\']', re.IGNORECASE),
            "token": re.compile(r'token["\']?\s*[:=]\s*["\']([^"\']{20,})["\']', re.IGNORECASE),
        }

        # Scan Python files
        for py_file in self.root_path.rglob("*.py"):
            # Skip test files and venv
            if "test" in str(py_file) or ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()
                    for secret_type, pattern in patterns.items():
                        matches = pattern.findall(content)
                        if matches:
                            secrets.append(
                                {
                                    "file": str(py_file.relative_to(self.root_path)),
                                    "type": secret_type,
                                    "severity": "HIGH",
                                }
                            )
            except Exception:
                continue

        return secrets

    def check_gitignore(self) -> Tuple[bool, List[str]]:
        """Check .gitignore for proper exclusions."""
        errors = []
        gitignore = self.root_path / ".gitignore"

        if not gitignore.exists():
            errors.append("❌ .gitignore not found")
            return False, errors

        required_patterns = [
            ".env",
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            "logs/",
            "data/raw/",
            "data/processed/",
            "models/",
            "signals/",
        ]

        with open(gitignore) as f:
            gitignore_content = f.read()

        for pattern in required_patterns:
            if pattern not in gitignore_content:
                errors.append(f"⚠️  Missing '{pattern}' in .gitignore")

        return len(errors) == 0, errors

    def validate_directory_structure(self) -> Tuple[bool, List[str]]:
        """Validate expected directory structure exists."""
        errors = []
        required_dirs = [
            "src/data_ingestion",
            "src/feature_engineering",
            "src/modeling",
            "src/strategy",
            "src/execution",
            "src/utils",
            "tests",
            "config",
            "docs",
        ]

        for dir_path in required_dirs:
            full_path = self.root_path / dir_path
            if not full_path.exists():
                errors.append(f"❌ Directory missing: {dir_path}")

        return len(errors) == 0, errors

    def validate_dependencies(self) -> Tuple[bool, List[str]]:
        """Validate requirements.txt exists and has key dependencies."""
        errors = []
        req_file = self.root_path / "requirements.txt"

        if not req_file.exists():
            errors.append("❌ requirements.txt not found")
            return False, errors

        with open(req_file) as f:
            requirements = f.read()

        critical_deps = [
            "pandas",
            "numpy",
            "scikit-learn",
            "yfinance",
        ]

        for dep in critical_deps:
            if dep not in requirements:
                errors.append(f"❌ Missing critical dependency: {dep}")

        return len(errors) == 0, errors

    def run_full_validation(self) -> Dict[str, any]:
        """Run all validation checks."""
        results = {
            "env_file": self.validate_env_file(),
            "settings_yaml": self.validate_settings_yaml(),
            "gitignore": self.check_gitignore(),
            "directory_structure": self.validate_directory_structure(),
            "dependencies": self.validate_dependencies(),
            "secrets_scan": self.scan_for_secrets(),
        }

        # Calculate overall status
        all_passed = all(result[0] for key, result in results.items() if key != "secrets_scan")
        no_secrets = len(results["secrets_scan"]) == 0

        results["overall_status"] = "PASS" if (all_passed and no_secrets) else "FAIL"
        return results


def print_validation_report(results: Dict[str, any]):
    """Print a formatted validation report."""
    print("\n" + "=" * 60)
    print("🔍 Configuration Validation Report")
    print("=" * 60 + "\n")

    for check, result in results.items():
        if check == "overall_status":
            continue

        if check == "secrets_scan":
            print(f"\n🔐 Secrets Scan:")
            if not result:
                print("  ✅ No hardcoded secrets detected")
            else:
                print(f"  ❌ Found {len(result)} potential secret(s):")
                for secret in result:
                    print(f"     - {secret['file']}: {secret['type']} ({secret['severity']})")
        else:
            passed, errors = result
            status = "✅" if passed else "❌"
            print(f"\n{status} {check.replace('_', ' ').title()}:")
            if errors:
                for error in errors:
                    print(f"   {error}")
            else:
                print("   All checks passed")

    print("\n" + "=" * 60)
    print(f"Overall Status: {results['overall_status']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    validator = ConfigValidator()
    results = validator.run_full_validation()
    print_validation_report(results)
