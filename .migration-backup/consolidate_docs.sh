#!/bin/bash
# Consolidate markdown files

echo "📁 Consolidating Documentation..."

# Create docs/ directory if it doesn't exist
mkdir -p docs/archive

# Keep these in root (essential)
KEEP_IN_ROOT=(
    "README.md"
    "CONTRIBUTING.md"
    "SECURITY.md"
    "PRODUCTION_UPGRADE_PLAN.md"
)

# Move completion reports to docs/
echo "Moving completion reports to docs/..."
mv ADVANCED_ML_COMPLETE.md docs/ 2>/dev/null
mv RL_AGENTS_COMPLETE.md docs/ 2>/dev/null
mv INTELLIGENCE_NETWORK_COMPLETE.md docs/ 2>/dev/null
mv BROKER_INTEGRATION_COMPLETE.md docs/ 2>/dev/null
mv INFRASTRUCTURE_COMPLETE.md docs/ 2>/dev/null
mv DASHBOARD_COMPLETION.md docs/ 2>/dev/null

# Move setup/deployment docs to docs/
echo "Moving setup guides to docs/..."
mv BASE_SETUP_GUIDE.md docs/ 2>/dev/null
mv DEPLOYMENT.md docs/ 2>/dev/null
mv GETTING_STARTED.md docs/ 2>/dev/null
mv QUICKSTART.md docs/ 2>/dev/null

# Move status/report files to docs/
echo "Moving status reports to docs/..."
mv STATUS-REPORT.md docs/ 2>/dev/null
mv DEPENDENCY_STATUS.md docs/ 2>/dev/null
mv SECURITY_RESOLUTION_REPORT.md docs/ 2>/dev/null
mv INTEGRATION_TEST_REPORT.md docs/ 2>/dev/null

# Archive meta files (cleanup summaries)
echo "Archiving meta files..."
mv MARKDOWN_CLEANUP_SUMMARY.md docs/archive/ 2>/dev/null
mv REORGANIZATION_SUMMARY.md docs/archive/ 2>/dev/null
mv FIXES.md docs/archive/ 2>/dev/null

# Count remaining files
ROOT_MD=$(ls -1 *.md 2>/dev/null | wc -l)
DOCS_MD=$(ls -1 docs/*.md 2>/dev/null | wc -l)

echo ""
echo "✅ Consolidation complete!"
echo "   Root markdown files: $ROOT_MD"
echo "   Docs markdown files: $DOCS_MD"
echo ""
echo "Files in root:"
ls -1 *.md 2>/dev/null
