#!/usr/bin/env python3
"""
Python 3.12 Compatibility Checker

Scans the codebase for Python 3.12 incompatibilities and deprecated features.
"""
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

# ANSI colors
RED = '\033[91m'
YELLOW = '\033[93m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'


class CompatibilityChecker:
    """Checks code for Python 3.12 compatibility issues."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.issues = defaultdict(list)
        self.file_count = 0
        self.line_count = 0

    def check_all(self):
        """Run all compatibility checks."""
        print("=" * 70)
        print("PYTHON 3.12 COMPATIBILITY CHECK")
        print("=" * 70)
        print(f"\nScanning directory: {self.root_dir}")
        print()

        # Find all Python files
        python_files = list(self.root_dir.rglob("*.py"))
        print(f"Found {len(python_files)} Python files")
        print()

        # Run checks
        for file_path in python_files:
            if self._should_skip(file_path):
                continue

            self.file_count += 1
            self._check_file(file_path)

        # Print results
        self._print_results()

    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_dirs = {'.venv', 'venv', '__pycache__', '.git', 'node_modules', 'build', 'dist'}
        return any(part in skip_dirs for part in file_path.parts)

    def _check_file(self, file_path: Path):
        """Check a single file for compatibility issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                self.line_count += len(lines)

                for line_num, line in enumerate(lines, 1):
                    self._check_line(file_path, line_num, line)

        except Exception as e:
            self.issues['errors'].append((file_path, 0, f"Error reading file: {e}"))

    def _check_line(self, file_path: Path, line_num: int, line: str):
        """Check a single line for compatibility issues."""

        # Check for removed modules
        removed_modules = {
            'distutils': 'Use setuptools or packaging instead',
            'imp ': 'Use importlib instead',
            'asyncore': 'Use asyncio instead',
            'asynchat': 'Use asyncio instead',
        }

        for module, suggestion in removed_modules.items():
            if re.search(rf'\bimport\s+{module}|from\s+{module}', line):
                self.issues['removed_modules'].append(
                    (file_path, line_num, f"Removed module '{module}': {suggestion}")
                )

        # Check for deprecated unittest methods
        if 'assertEquals' in line or 'assertNotEquals' in line:
            self.issues['deprecated_methods'].append(
                (file_path, line_num, "Use assertEqual/assertNotEqual instead")
            )

        # Check for old-style string formatting (not removed, but discouraged)
        if re.search(r'%[sd]', line) and 'format' not in line:
            self.issues['old_string_format'].append(
                (file_path, line_num, "Consider using f-strings or .format()")
            )

        # Check for typing imports that can be simplified in 3.9+
        typing_imports = [
            ('typing.List', 'list'),
            ('typing.Dict', 'dict'),
            ('typing.Set', 'set'),
            ('typing.Tuple', 'tuple'),
            ('typing.Optional', '| None'),
        ]

        for old, new in typing_imports:
            if old in line:
                self.issues['outdated_typing'].append(
                    (file_path, line_num, f"Can use built-in '{new}' instead of '{old}'")
                )

        # Check for Exception chaining that can be improved
        if re.search(r'raise\s+\w+\(.*\)\s+from\s+None', line):
            self.issues['exception_chaining'].append(
                (file_path, line_num, "Suppressing exception context - ensure this is intentional")
            )

        # Check for deprecated ssl module constants
        if 'ssl.PROTOCOL_TLS' in line or 'ssl.PROTOCOL_SSLv' in line:
            self.issues['ssl_deprecated'].append(
                (file_path, line_num, "Use ssl.PROTOCOL_TLS_CLIENT or ssl.PROTOCOL_TLS_SERVER")
            )

    def _print_results(self):
        """Print compatibility check results."""
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\nFiles scanned: {self.file_count}")
        print(f"Lines scanned: {self.line_count:,}")
        print()

        if not self.issues:
            print(f"{GREEN}✅ No compatibility issues found!{RESET}")
            print()
            print("Your code appears to be Python 3.12 compatible.")
            return

        # Print issues by category
        total_issues = sum(len(issues) for issues in self.issues.values())
        print(f"{YELLOW}⚠️  Found {total_issues} potential issues:{RESET}\n")

        # Critical issues (removed modules)
        if 'removed_modules' in self.issues:
            self._print_issue_category(
                "🚨 CRITICAL - Removed Modules",
                self.issues['removed_modules'],
                RED
            )

        # High priority (deprecated methods)
        if 'deprecated_methods' in self.issues:
            self._print_issue_category(
                "⚠️  HIGH - Deprecated Methods",
                self.issues['deprecated_methods'],
                YELLOW
            )

        # Medium priority (typing)
        if 'outdated_typing' in self.issues:
            self._print_issue_category(
                "ℹ️  MEDIUM - Outdated Type Hints",
                self.issues['outdated_typing'],
                BLUE
            )

        # Low priority (style)
        if 'old_string_format' in self.issues:
            self._print_issue_category(
                "💡 LOW - Old String Formatting",
                self.issues['old_string_format'],
                BLUE,
                max_display=5
            )

        # Other issues
        for category in ['ssl_deprecated', 'exception_chaining', 'errors']:
            if category in self.issues:
                self._print_issue_category(
                    f"ℹ️  {category.replace('_', ' ').title()}",
                    self.issues[category],
                    YELLOW
                )

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        critical = len(self.issues.get('removed_modules', []))
        high = len(self.issues.get('deprecated_methods', []))
        medium = len(self.issues.get('outdated_typing', []))
        low = total_issues - critical - high - medium

        print(f"\n{RED}Critical:{RESET} {critical}")
        print(f"{YELLOW}High:{RESET}     {high}")
        print(f"{BLUE}Medium:{RESET}   {medium}")
        print(f"{BLUE}Low:{RESET}      {low}")
        print(f"\n{YELLOW}Total:{RESET}    {total_issues}")

        # Recommendations
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        print()

        if critical > 0:
            print(f"{RED}⚠️  CRITICAL ISSUES FOUND{RESET}")
            print("These MUST be fixed before Python 3.12 upgrade:")
            print("  1. Replace removed modules (distutils, imp, asyncore)")
            print("  2. Test thoroughly after changes")
            print()

        if high > 0:
            print(f"{YELLOW}⚠️  HIGH PRIORITY ISSUES{RESET}")
            print("Should be fixed soon:")
            print("  1. Update deprecated unittest methods")
            print("  2. Review SSL protocol usage")
            print()

        if medium > 0 or low > 0:
            print(f"{BLUE}ℹ️  OPTIONAL IMPROVEMENTS{RESET}")
            print("Consider modernizing:")
            print("  1. Use built-in types for type hints (Python 3.9+)")
            print("  2. Use f-strings for better readability")
            print()

        # Exit code
        sys.exit(1 if critical > 0 or high > 0 else 0)

    def _print_issue_category(
        self,
        title: str,
        issues: List[Tuple],
        color: str,
        max_display: int = 10
    ):
        """Print issues for a specific category."""
        print(f"\n{color}{title}{RESET} ({len(issues)} issues)")
        print("-" * 70)

        for i, (file_path, line_num, message) in enumerate(issues[:max_display], 1):
            rel_path = file_path.relative_to(self.root_dir)
            print(f"{i}. {rel_path}:{line_num}")
            print(f"   {message}")

        if len(issues) > max_display:
            print(f"\n   ... and {len(issues) - max_display} more")

        print()


def main():
    """Main entry point."""
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Check if src directory exists
    src_dir = project_root / "src"
    if not src_dir.exists():
        print(f"Error: src directory not found at {src_dir}")
        sys.exit(1)

    # Run compatibility check
    checker = CompatibilityChecker(src_dir)
    checker.check_all()


if __name__ == "__main__":
    main()
