#!/usr/bin/env python3
"""
Convert Print Statements to Logging
====================================

Automatically converts print() statements to appropriate logger calls.

Usage:
    python scripts/convert_prints_to_logging.py src/
    python scripts/convert_prints_to_logging.py src/ --dry-run
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Tuple

def has_logging_setup(content: str) -> Tuple[bool, bool]:
    """Check if file has logging import and logger instance."""
    has_import = bool(re.search(r'^import logging', content, re.MULTILINE))
    has_logger = bool(re.search(r'^logger = logging\.getLogger', content, re.MULTILINE))
    return has_import, has_logger

def add_logging_setup(content: str) -> str:
    """Add logging import and logger instance if missing."""
    has_import, has_logger = has_logging_setup(content)

    lines = content.split('\n')

    # Find end of imports block
    import_end_idx = 0
    for i, line in enumerate(lines):
        if line.strip() and (line.startswith('import ') or line.startswith('from ')):
            import_end_idx = i + 1

    # Add logging import if missing
    if not has_import:
        if import_end_idx > 0:
            lines.insert(import_end_idx, 'import logging')
            import_end_idx += 1
        else:
            # No imports, add at top after docstring
            docstring_end = 0
            if lines[0].strip().startswith('"""') or lines[0].strip().startswith("'''"):
                for i in range(1, len(lines)):
                    if '"""' in lines[i] or "'''" in lines[i]:
                        docstring_end = i + 1
                        break
            lines.insert(docstring_end, 'import logging')
            lines.insert(docstring_end + 1, '')
            import_end_idx = docstring_end + 2

    # Add logger instance if missing
    if not has_logger:
        if import_end_idx < len(lines):
            lines.insert(import_end_idx, '')
            lines.insert(import_end_idx + 1, 'logger = logging.getLogger(__name__)')
            lines.insert(import_end_idx + 2, '')

    return '\n'.join(lines)

def convert_prints_to_logging(content: str) -> Tuple[str, int]:
    """Convert print statements to logger calls."""
    conversions = 0

    # Pattern 1: print("simple string")
    pattern1 = r'\bprint\s*\(\s*"([^"]*)"\s*\)'
    content, count1 = re.subn(pattern1, r'logger.info("\1")', content)
    conversions += count1

    # Pattern 2: print('simple string')
    pattern2 = r"\bprint\s*\(\s*'([^']*)'\s*\)"
    content, count2 = re.subn(pattern2, r"logger.info('\1')", content)
    conversions += count2

    # Pattern 3: print(f"formatted {var}")
    pattern3 = r'\bprint\s*\(\s*f"([^"]*)"\s*\)'
    content, count3 = re.subn(pattern3, r'logger.info(f"\1")', content)
    conversions += count3

    # Pattern 4: print(f'formatted {var}')
    pattern4 = r"\bprint\s*\(\s*f'([^']*)'\s*\)"
    content, count4 = re.subn(pattern4, r"logger.info(f'\1')", content)
    conversions += count4

    # Pattern 5: print(variable)
    # This one is trickier - might need context to determine log level
    # For now, default to info
    pattern5 = r'\bprint\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)'
    content, count5 = re.subn(pattern5, r'logger.info(\1)', content)
    conversions += count5

    return content, conversions

def process_file(file_path: Path, dry_run: bool = False) -> Tuple[int, str]:
    """Process a single Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # Check if file has print statements
        if 'print(' not in original_content:
            return 0, "No print statements"

        # Add logging setup if needed
        content = add_logging_setup(original_content)

        # Convert print statements
        content, conversions = convert_prints_to_logging(content)

        if conversions > 0 and not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return conversions, "Modified"
        elif conversions > 0 and dry_run:
            return conversions, "Would modify"
        else:
            return 0, "No conversions"

    except Exception as e:
        return 0, f"Error: {e}"

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Convert print() statements to logging calls'
    )
    parser.add_argument(
        'directory',
        type=str,
        help='Directory to process'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    parser.add_argument(
        '--exclude',
        type=str,
        nargs='+',
        default=['tests', 'examples', 'demo'],
        help='Directories to exclude'
    )

    args = parser.parse_args()

    # Validate directory
    directory = Path(args.directory)
    if not directory.exists():
        print(f"❌ Error: Directory '{directory}' does not exist")
        sys.exit(1)

    print(f"🔍 Scanning Python files in {directory}...")
    if args.dry_run:
        print("   (DRY RUN - no files will be modified)")

    # Process files
    total_conversions = 0
    files_modified = 0
    files_processed = 0

    for file_path in directory.rglob('*.py'):
        # Skip excluded directories
        if any(excluded in file_path.parts for excluded in args.exclude):
            continue

        conversions, status = process_file(file_path, args.dry_run)
        files_processed += 1

        if conversions > 0:
            files_modified += 1
            total_conversions += conversions
            symbol = '○' if args.dry_run else '✓'
            print(f"{symbol} {file_path.relative_to(directory)}: {conversions} conversions ({status})")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"📊 Summary:")
    print(f"   Files processed: {files_processed}")
    print(f"   Files modified: {files_modified}")
    print(f"   Total conversions: {total_conversions}")

    if args.dry_run and files_modified > 0:
        print(f"\n💡 Run without --dry-run to apply changes")

    return 0 if total_conversions >= 0 else 1

if __name__ == '__main__':
    sys.exit(main())
