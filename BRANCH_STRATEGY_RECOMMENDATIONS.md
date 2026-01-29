# Branch Strategy Recommendations

This document provides additional recommendations for implementing and enforcing the branch management strategy.

## 🛡️ GitHub Branch Protection Settings

### Recommended Settings for `main` Branch

Navigate to: **Settings → Branches → Add rule**

```yaml
Branch name pattern: main

Protection rules:
✅ Require a pull request before merging
  ✅ Require approvals: 1
  ✅ Dismiss stale pull request approvals when new commits are pushed
  ✅ Require review from Code Owners (if CODEOWNERS file exists)
  
✅ Require status checks to pass before merging
  ✅ Require branches to be up to date before merging
  Status checks that are required:
    - ci (GitHub Actions)
    - tests
    - lint
    
✅ Require conversation resolution before merging

✅ Require signed commits (optional, recommended for security)

✅ Require linear history (optional, enforces rebase/squash)

✅ Include administrators (recommended to enforce rules on all)

✅ Restrict who can push to matching branches
  - Only: Maintainers, admins
  
✅ Allow force pushes: Never

✅ Allow deletions: Never
```

### Recommended Settings for `develop` Branch

```yaml
Branch name pattern: develop

Protection rules:
✅ Require a pull request before merging
  ✅ Require approvals: 1 (can be 0 for faster iteration)
  
✅ Require status checks to pass before merging
  Status checks that are required:
    - ci
    - tests
    - lint
    
✅ Require conversation resolution before merging

✅ Allow force pushes: Administrators only

✅ Allow deletions: Never
```

## 📊 Automated Branch Management

### GitHub Actions Workflow: Branch Cleanup

Create `.github/workflows/branch-cleanup.yml`:

```yaml
name: Branch Cleanup

on:
  schedule:
    # Run weekly on Sundays at midnight
    - cron: '0 0 * * 0'
  workflow_dispatch:  # Allow manual trigger

jobs:
  cleanup:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Fetch all history
          
      - name: Delete merged branches
        run: |
          # Delete remote branches that are merged to develop
          git fetch --prune
          
          # Get list of merged branches (exclude main, develop, and current branch)
          merged_branches=$(git branch -r --merged origin/develop | \
            grep -v "HEAD" | \
            grep -v "origin/main" | \
            grep -v "origin/develop" | \
            sed 's/origin\///')
          
          echo "Branches to delete:"
          echo "$merged_branches"
          
          # Delete each branch
          for branch in $merged_branches; do
            if [ ! -z "$branch" ]; then
              echo "Deleting branch: $branch"
              git push origin --delete "$branch" || echo "Failed to delete $branch"
            fi
          done
          
      - name: Report stale branches
        run: |
          # Find branches older than 90 days
          echo "Stale branches (>90 days, not merged):"
          git for-each-ref --sort=-committerdate refs/remotes/ \
            --format='%(committerdate:short) %(refname:short)' | \
            awk -v d="$(date -d '90 days ago' +%Y-%m-%d)" '$1 < d' | \
            grep -v "origin/main" | \
            grep -v "origin/develop"
```

### GitHub Actions Workflow: Branch Naming Validation

Create `.github/workflows/branch-naming.yml`:

```yaml
name: Branch Naming Validation

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  validate-branch-name:
    runs-on: ubuntu-latest
    
    steps:
      - name: Check branch naming convention
        run: |
          BRANCH_NAME="${{ github.head_ref }}"
          
          # Define valid patterns
          FEATURE_PATTERN="^feature/phase[0-9]+-[a-z0-9-]+$"
          BUGFIX_PATTERN="^bugfix/[0-9]+-[a-z0-9-]+$"
          HOTFIX_PATTERN="^hotfix/[0-9]+\.[0-9]+\.[0-9]+-[a-z0-9-]+$"
          RELEASE_PATTERN="^release/[0-9]+\.[0-9]+\.[0-9]+$"
          PHASE_PATTERN="^phase/[0-9]+-[a-z0-9-]+$"
          COPILOT_PATTERN="^copilot/.*$"  # Allow copilot branches
          
          if [[ $BRANCH_NAME =~ $FEATURE_PATTERN ]] || \
             [[ $BRANCH_NAME =~ $BUGFIX_PATTERN ]] || \
             [[ $BRANCH_NAME =~ $HOTFIX_PATTERN ]] || \
             [[ $BRANCH_NAME =~ $RELEASE_PATTERN ]] || \
             [[ $BRANCH_NAME =~ $PHASE_PATTERN ]] || \
             [[ $BRANCH_NAME =~ $COPILOT_PATTERN ]]; then
            echo "✅ Branch name '$BRANCH_NAME' follows naming convention"
          else
            echo "❌ Branch name '$BRANCH_NAME' does not follow naming convention"
            echo ""
            echo "Valid patterns:"
            echo "  - feature/phase<N>-<description>"
            echo "  - bugfix/<issue-number>-<description>"
            echo "  - hotfix/<version>-<description>"
            echo "  - release/<version>"
            echo "  - phase/<N>-<description>"
            echo ""
            echo "Examples:"
            echo "  - feature/phase3-reddit-sentiment"
            echo "  - bugfix/123-fix-api-error"
            echo "  - hotfix/1.2.1-security-patch"
            echo "  - release/2.0.0"
            echo ""
            echo "See BRANCH_MANAGEMENT.md for details"
            exit 1
          fi
```

### GitHub Actions Workflow: PR Title Validation

Create `.github/workflows/pr-title-check.yml`:

```yaml
name: PR Title Check

on:
  pull_request:
    types: [opened, edited, synchronize]

jobs:
  check-pr-title:
    runs-on: ubuntu-latest
    
    steps:
      - name: Validate PR title format
        run: |
          PR_TITLE="${{ github.event.pull_request.title }}"
          
          # Expected format: [Phase <N>] <type>: <description>
          PHASE_PATTERN="^\[Phase [0-9]+\] (feat|fix|docs|style|refactor|test|chore|perf|ci):"
          INFRA_PATTERN="^\[Infrastructure\] (feat|fix|docs|style|refactor|test|chore|perf|ci):"
          OTHER_PATTERN="^\[.*\] (feat|fix|docs|style|refactor|test|chore|perf|ci):"
          
          if [[ $PR_TITLE =~ $PHASE_PATTERN ]] || \
             [[ $PR_TITLE =~ $INFRA_PATTERN ]] || \
             [[ $PR_TITLE =~ $OTHER_PATTERN ]]; then
            echo "✅ PR title follows convention: $PR_TITLE"
          else
            echo "❌ PR title does not follow convention: $PR_TITLE"
            echo ""
            echo "Expected format: [Phase <N>] <type>: <description>"
            echo ""
            echo "Valid types: feat, fix, docs, style, refactor, test, chore, perf, ci"
            echo ""
            echo "Examples:"
            echo "  - [Phase 3] feat: Add Reddit sentiment analysis"
            echo "  - [Phase 2] fix: Correct order execution logic"
            echo "  - [Infrastructure] chore: Update dependencies"
            echo ""
            exit 1
          fi
```

## 🏷️ Recommended Labels

### Setting Up Labels in GitHub

Navigate to: **Issues → Labels**

#### Type Labels
```
bug               - #d73a4a - Something isn't working
enhancement       - #a2eeef - New feature or request
documentation     - #0075ca - Improvements to documentation
question          - #d876e3 - Further information requested
help-wanted       - #008672 - Extra attention needed
good-first-issue  - #7057ff - Good for newcomers
```

#### Phase Labels
```
phase-1  - #1d76db - Core Trading System
phase-2  - #1d76db - Broker Integration
phase-3  - #1d76db - Intelligence Network
phase-4  - #1d76db - Deep Learning
phase-5  - #1d76db - RL Agents
phase-6  - #1d76db - Dashboard
phase-7  - #1d76db - Infrastructure
phase-8+ - #1d76db - Future Phases
```

#### Priority Labels
```
priority-critical - #b60205 - Critical issues
priority-high     - #d93f0b - High priority
priority-medium   - #fbca04 - Medium priority
priority-low      - #0e8a16 - Low priority
```

#### Status Labels
```
in-progress - #fbca04 - Currently being worked on
blocked     - #d73a4a - Waiting on dependencies
needs-review - #a2eeef - Ready for review
wontfix     - #ffffff - Will not be addressed
duplicate   - #cfd3d7 - Duplicate issue
invalid     - #e4e669 - Invalid issue
```

## 📝 CODEOWNERS File

Create `.github/CODEOWNERS`:

```
# Default owners for everything
* @cpoplaws

# Documentation
*.md @cpoplaws
/docs/ @cpoplaws

# CI/CD
/.github/ @cpoplaws

# Core trading system (Phase 1)
/src/data_ingestion/ @cpoplaws
/src/feature_engineering/ @cpoplaws
/src/modeling/ @cpoplaws

# Advanced features
/src/advanced_strategies/ @cpoplaws

# Infrastructure
/docker-compose.yml @cpoplaws
/Dockerfile @cpoplaws
/Makefile @cpoplaws

# Configuration
/config/ @cpoplaws
.env.* @cpoplaws
```

## 🔄 Git Hooks with pre-commit

Enhance `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: check-json
      - id: pretty-format-json
        args: ['--autofix']
      - id: check-case-conflict
      - id: detect-private-key
      
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.11
        
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.11
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
        
  - repo: local
    hooks:
      - id: branch-naming
        name: Check branch naming
        entry: python .github/scripts/check_branch_name.py
        language: python
        pass_filenames: false
        always_run: true
```

Create `.github/scripts/check_branch_name.py`:

```python
#!/usr/bin/env python3
"""Check if current branch follows naming convention."""
import re
import subprocess
import sys


def get_current_branch():
    """Get current git branch name."""
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()


def check_branch_name(branch: str) -> bool:
    """Check if branch name follows convention."""
    patterns = [
        r"^feature/phase\d+-[a-z0-9-]+$",
        r"^bugfix/\d+-[a-z0-9-]+$",
        r"^hotfix/\d+\.\d+\.\d+-[a-z0-9-]+$",
        r"^release/\d+\.\d+\.\d+$",
        r"^phase/\d+-[a-z0-9-]+$",
        r"^main$",
        r"^develop$",
        r"^copilot/.*$",  # Allow copilot branches
    ]
    
    return any(re.match(pattern, branch) for pattern in patterns)


def main():
    """Main entry point."""
    branch = get_current_branch()
    
    # Skip check for main/develop
    if branch in ["main", "develop"]:
        sys.exit(0)
    
    if not check_branch_name(branch):
        print(f"❌ Branch name '{branch}' does not follow naming convention")
        print()
        print("Valid patterns:")
        print("  - feature/phase<N>-<description>")
        print("  - bugfix/<issue-number>-<description>")
        print("  - hotfix/<version>-<description>")
        print("  - release/<version>")
        print()
        print("See BRANCH_MANAGEMENT.md for details")
        sys.exit(1)
    
    print(f"✅ Branch name '{branch}' follows convention")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

## 📊 Monitoring & Metrics

### Metrics to Track

Create a monthly dashboard to track:

1. **Branch Health**
   - Number of active branches
   - Average branch lifetime
   - Stale branches (>90 days)
   - Branches awaiting merge

2. **PR Metrics**
   - PR merge time (time from open to merge)
   - Review time
   - PR size distribution
   - PR rejection rate

3. **Code Quality**
   - Test coverage trend
   - Linting issues
   - Security vulnerabilities
   - Documentation coverage

4. **Contribution Metrics**
   - Active contributors
   - First-time contributors
   - Contribution distribution by phase

### Quarterly Review Checklist

- [ ] Review stale branches (>90 days)
- [ ] Archive completed phase branches
- [ ] Update branch protection rules if needed
- [ ] Review and improve PR template
- [ ] Update issue templates based on feedback
- [ ] Audit CODEOWNERS assignments
- [ ] Review CI/CD workflows
- [ ] Update documentation

## 🎯 Implementation Timeline

### Week 1: Setup Foundation
- [ ] Configure branch protection for `main` and `develop`
- [ ] Add GitHub labels
- [ ] Create CODEOWNERS file
- [ ] Merge this PR with documentation

### Week 2: Automation
- [ ] Add branch cleanup workflow
- [ ] Add branch naming validation
- [ ] Add PR title check workflow
- [ ] Update pre-commit hooks

### Week 3: Training & Adoption
- [ ] Share documentation with team
- [ ] Create tutorial video (optional)
- [ ] Answer questions and clarify workflow
- [ ] Monitor first few PRs for compliance

### Week 4: Refinement
- [ ] Collect feedback
- [ ] Adjust templates and workflows
- [ ] Document lessons learned
- [ ] Plan next improvements

## 📚 Additional Resources

- [GitHub Branch Protection Docs](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [CODEOWNERS Documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- [Pre-commit Hooks](https://pre-commit.com/)

---

**Remember**: Start simple and iterate. Don't try to enforce everything at once. Build team habits gradually! 🚀
