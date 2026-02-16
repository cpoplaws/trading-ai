#!/usr/bin/env python3
"""
Add Module Docstrings to __init__.py Files
===========================================

Automatically adds descriptive docstrings to __init__.py files that are missing them.

Usage:
    python scripts/add_module_docstrings.py
    python scripts/add_module_docstrings.py --dry-run
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

# Module descriptions (can be customized)
MODULE_DESCRIPTIONS: Dict[str, str] = {
    'database': 'Database connectivity, ORM models, and data persistence',
    'backtesting': 'Backtesting engine and strategy evaluation',
    'exchanges': 'Exchange API clients and trading interfaces',
    'utils': 'Utility functions and helper modules',
    'modeling': 'Machine learning models and training',
    'blockchain': 'Blockchain interactions and on-chain analysis',
    'execution': 'Trade execution and order management',
    'autonomous_agent': 'Autonomous trading agent framework',
    'feature_engineering': 'Feature extraction and engineering for ML',
    'defi': 'DeFi protocol integrations and strategies',
    'data_ingestion': 'Data collection and ingestion pipelines',
    'strategy': 'Trading strategy implementations',
    'crypto_strategies': 'Cryptocurrency trading strategies',
    'advanced_strategies': 'Advanced and complex trading strategies',
    'ml': 'Machine learning and predictive models',
    'risk_management': 'Risk management and position sizing',
    'monitoring': 'System monitoring and dashboards',
    'api': 'REST API endpoints and routes',
    'crypto_data': 'Cryptocurrency data collection and processing',
    'onchain': 'On-chain data analysis and metrics',
    'data_collection': 'External data source collectors',
    'dex': 'Decentralized exchange integrations',
    'alerts': 'Alert and notification system',
    'crypto_ml': 'Machine learning for crypto trading',
    'infrastructure': 'Infrastructure components and services',
}

DOCSTRING_TEMPLATE = '''"""
{module_name}
{"=" * len(module_name)}

{description}

{components}

Usage:
    >>> from src.{module_path} import {example_import}

See Also:
    - docs/ARCHITECTURE.md for system architecture
    - docs/API_REFERENCE.md for API documentation
"""
'''

def get_module_name(module_path: Path) -> str:
    """Get human-readable module name from path."""
    name = module_path.name
    # Convert snake_case to Title Case
    return ' '.join(word.capitalize() for word in name.split('_'))

def get_module_description(module_path: Path) -> str:
    """Get module description from predefined dict or generate default."""
    module_key = module_path.name
    if module_key in MODULE_DESCRIPTIONS:
        return MODULE_DESCRIPTIONS[module_key]
    else:
        return f"Core functionality for {get_module_name(module_path).lower()}"

def get_components_section(init_file: Path) -> str:
    """Extract components from __init__.py imports."""
    try:
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Look for imports
        components = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('from .') and 'import' in line:
                # Extract imported names
                parts = line.split('import')
                if len(parts) > 1:
                    imports = parts[1].strip()
                    if '(' not in imports:  # Skip multi-line imports for simplicity
                        for item in imports.split(','):
                            item = item.strip()
                            if item:
                                components.append(f"    - {item}")

        if components:
            return "Key Components:\n" + '\n'.join(components[:5])  # Limit to 5
        else:
            return "This module provides core functionality for the Trading AI system."

    except Exception:
        return "This module provides core functionality for the Trading AI system."

def get_example_import(init_file: Path) -> str:
    """Get an example import from the module."""
    try:
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Look for first import
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('from .') and 'import' in line:
                parts = line.split('import')
                if len(parts) > 1:
                    imports = parts[1].strip()
                    first_import = imports.split(',')[0].strip()
                    if first_import and '(' not in first_import:
                        return first_import

        return "Component"

    except Exception:
        return "Component"

def has_docstring(init_file: Path) -> bool:
    """Check if __init__.py already has a docstring."""
    try:
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Check if file starts with docstring
        return content.startswith('"""') or content.startswith("'''")

    except Exception:
        return False

def add_docstring(init_file: Path, dry_run: bool = False) -> tuple[bool, str]:
    """Add docstring to __init__.py file."""

    # Check if already has docstring
    if has_docstring(init_file):
        return False, "Already has docstring"

    try:
        # Read existing content
        with open(init_file, 'r', encoding='utf-8') as f:
            existing_content = f.read()

        # Get module info
        module_path = init_file.parent.relative_to('src')
        module_name = get_module_name(module_path)
        description = get_module_description(module_path)
        components = get_components_section(init_file)
        example_import = get_example_import(init_file)

        # Create docstring
        docstring = DOCSTRING_TEMPLATE.format(
            module_name=module_name,
            description=description,
            components=components,
            module_path=str(module_path).replace('/', '.'),
            example_import=example_import
        )

        # Combine with existing content
        new_content = docstring + '\n' + existing_content

        # Write back if not dry run
        if not dry_run:
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True, "Docstring added"
        else:
            return True, "Would add docstring"

    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Add module docstrings to __init__.py files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    parser.add_argument(
        '--directory',
        type=str,
        default='src',
        help='Directory to process (default: src)'
    )

    args = parser.parse_args()

    # Validate directory
    directory = Path(args.directory)
    if not directory.exists():
        print(f"❌ Error: Directory '{directory}' does not exist")
        sys.exit(1)

    print(f"🔍 Processing __init__.py files in {directory}...")
    if args.dry_run:
        print("   (DRY RUN - no files will be modified)")

    # Process files
    files_modified = 0
    files_skipped = 0
    files_errored = 0

    for init_file in directory.rglob('__init__.py'):
        modified, status = add_docstring(init_file, args.dry_run)

        if modified:
            files_modified += 1
            symbol = '○' if args.dry_run else '✓'
            rel_path = init_file.relative_to(directory.parent)
            print(f"{symbol} {rel_path}: {status}")
        elif "Error" in status:
            files_errored += 1
            print(f"✗ {init_file.relative_to(directory.parent)}: {status}")
        else:
            files_skipped += 1
            # Don't print for skipped files (too verbose)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"📊 Summary:")
    print(f"   Files modified: {files_modified}")
    print(f"   Files skipped (already have docstrings): {files_skipped}")
    if files_errored > 0:
        print(f"   Files with errors: {files_errored}")

    if args.dry_run and files_modified > 0:
        print(f"\n💡 Run without --dry-run to apply changes")

    return 0 if files_errored == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
