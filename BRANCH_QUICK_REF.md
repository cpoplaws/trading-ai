# 🚀 Quick Branch Reference

Quick reference card for Trading-AI branching workflow.

## 🌿 Branch Types

| Type | Format | Branch From | Merge To | Example |
|:-----|:-------|:------------|:---------|:--------|
| **Feature** | `feature/phase<N>-<desc>` | `develop` | `develop` | `feature/phase3-reddit-sentiment` |
| **Bugfix** | `bugfix/<issue>-<desc>` | `develop` | `develop` | `bugfix/123-fix-api-error` |
| **Hotfix** | `hotfix/<ver>-<desc>` | `main` | `main` + `develop` | `hotfix/1.2.1-critical-fix` |
| **Release** | `release/<version>` | `develop` | `main` + `develop` | `release/2.0.0` |

## 📝 Quick Commands

### Start New Feature
```bash
git checkout develop
git pull origin develop
git checkout -b feature/phase3-new-feature
# ... make changes ...
git add .
git commit -m "feat(phase3): add new feature"
git push origin feature/phase3-new-feature
# Create PR to develop via GitHub
```

### Fix a Bug
```bash
git checkout develop
git pull origin develop
git checkout -b bugfix/456-fix-issue
# ... fix bug ...
git add .
git commit -m "fix: resolve issue #456"
git push origin bugfix/456-fix-issue
# Create PR to develop via GitHub
```

### Critical Hotfix
```bash
git checkout main
git pull origin main
git checkout -b hotfix/1.2.1-urgent-fix
# ... fix critical issue ...
git add .
git commit -m "fix: critical security patch"
git push origin hotfix/1.2.1-urgent-fix
# Create PR to main via GitHub
# After merge to main, also merge to develop
```

### Keep Branch Updated
```bash
git fetch origin
git rebase origin/develop
# Or if you prefer merge:
# git merge origin/develop
```

## 💬 Commit Message Format

```
<type>(<scope>): <subject>

[optional body]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Tests
- `chore`: Maintenance

### Examples
```bash
feat(phase3): add Reddit sentiment analysis
fix(backtest): correct equity curve calculation
docs(readme): update setup instructions
chore(deps): upgrade streamlit to 1.30.0
```

## 🏷️ Project Phases

- **Phase 1**: Core Trading System (✅ Complete)
- **Phase 2**: Broker Integration (70%)
- **Phase 3**: Intelligence Network (60%)
- **Phase 4**: Deep Learning (40%)
- **Phase 5**: RL Agents (Planned)
- **Phase 6**: Dashboard (85%)
- **Phase 7**: Infrastructure (25%)
- **Phase 8-12**: Future Research

## 🔄 PR Workflow

1. **Create branch** from `develop`
2. **Make changes** with clear commits
3. **Test thoroughly** (run tests, linting)
4. **Push to remote** and create PR
5. **Address review** feedback
6. **Merge to develop** when approved
7. **Delete branch** after merge

## ✅ Pre-PR Checklist

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/

# Test
pytest tests/ -v

# Review changes
git diff develop..your-branch
```

## 📋 PR Title Format

```
[Phase <N>] <type>: <description>
```

Examples:
- `[Phase 3] feat: Add Reddit sentiment`
- `[Phase 2] fix: Order execution bug`
- `[Infra] chore: Update dependencies`

## 🆘 Common Issues

### Branch out of date
```bash
git checkout your-branch
git fetch origin
git rebase origin/develop
# Resolve conflicts if any
git push --force-with-lease
```

### Wrong branch name
```bash
git branch -m old-name new-name
git push origin --delete old-name
git push origin new-name
```

### Committed to wrong branch
```bash
# On wrong branch
git log  # Copy commit SHA
git checkout correct-branch
git cherry-pick <commit-SHA>
```

## 🔗 Full Documentation

For complete details, see:
- [BRANCH_MANAGEMENT.md](BRANCH_MANAGEMENT.md) - Complete guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines

---

**Quick tip**: Bookmark this page for easy reference! 📌
