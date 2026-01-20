# 🌿 Branch Management Guide

## Overview

This document defines the branching strategy for the Trading-AI project. Our multi-phase development approach (Phases 1-12) requires a clear and maintainable branch structure to manage complexity effectively.

## 🎯 Branch Structure

### Main Branches

#### `main` (Production Branch)
- **Purpose**: Production-ready code only
- **Protection**: ✅ Protected (requires PR + reviews)
- **Deployment**: Auto-deploys to production
- **Stability**: Must always be deployable
- **Merges from**: `develop` only (via release PRs)

#### `develop` (Integration Branch)
- **Purpose**: Integration branch for features
- **Protection**: ✅ Protected (requires PR)
- **Quality**: All tests must pass
- **Merges from**: Feature branches, hotfix branches
- **Merges to**: `main` (for releases)

### Development Branches

#### Feature Branches: `feature/<phase>-<description>`
Format: `feature/phase<N>-<short-description>`

**Examples:**
- `feature/phase2-alpaca-integration`
- `feature/phase3-sentiment-analysis`
- `feature/phase4-transformer-models`
- `feature/phase5-rl-agents`
- `feature/phase6-dashboard-improvements`

**Guidelines:**
- Branch from: `develop`
- Merge to: `develop`
- Lifetime: Delete after merge
- Naming: Use kebab-case, be descriptive but concise
- One feature per branch

#### Bugfix Branches: `bugfix/<issue>-<description>`
Format: `bugfix/<issue-number>-<short-description>`

**Examples:**
- `bugfix/123-fix-data-fetch-error`
- `bugfix/456-correct-indicator-calculation`

**Guidelines:**
- Branch from: `develop`
- Merge to: `develop`
- Reference issue number when available

#### Hotfix Branches: `hotfix/<version>-<description>`
Format: `hotfix/<version>-<critical-issue>`

**Examples:**
- `hotfix/1.2.1-security-vulnerability`
- `hotfix/1.3.1-critical-api-failure`

**Guidelines:**
- Branch from: `main`
- Merge to: `main` AND `develop`
- Deploy immediately after merge
- Used only for critical production issues

#### Release Branches: `release/<version>`
Format: `release/<major>.<minor>.<patch>`

**Examples:**
- `release/1.0.0`
- `release/2.1.0`

**Guidelines:**
- Branch from: `develop`
- Merge to: `main` AND `develop`
- Only bug fixes, documentation, and release prep
- No new features
- Tag after merge to `main`

### Phase-Specific Long-Running Branches (Optional)

For major experimental features that span multiple sprints:

#### `phase/<N>-<name>`
Format: `phase/<number>-<phase-name>`

**Examples:**
- `phase/5-reinforcement-learning`
- `phase/8-quantum-ml-research`

**Guidelines:**
- Use sparingly for long-term experimental work
- Regular merges from `develop` to stay current
- Eventually merge back to `develop` when stable
- Delete after completion

## 📋 Workflow by Project Phase

### Phase 1: Core Trading System (✅ Complete)
- Status: Merged to `main`
- Branches: Archived

### Phase 2: Broker Integration (70% Complete)
- **Current work**: `feature/phase2-*`
- **Focus**: Alpaca integration, order management
- **Merge to**: `develop` when stable

### Phase 3: Intelligence Network (60% Complete)
- **Current work**: `feature/phase3-*`
- **Focus**: Real APIs (NewsAPI, Reddit, FRED)
- **Merge to**: `develop` when stable

### Phase 4: Deep Learning (40% Complete)
- **Upcoming**: `feature/phase4-transformers`
- **Focus**: TimesNet, Autoformer, Informer
- **Status**: Planned

### Phase 5: Reinforcement Learning (0% - Planned)
- **Future**: `phase/5-reinforcement-learning` (long-running)
- **Focus**: PPO/DDPG agents, custom trading env
- **Status**: Not started

### Phases 6-12: Future Development
- Follow feature branch pattern
- Create phase-specific branches for experimental work
- Merge incrementally to `develop`

## 🔄 Branch Workflow

### Creating a New Feature

```bash
# 1. Start from develop
git checkout develop
git pull origin develop

# 2. Create feature branch
git checkout -b feature/phase3-reddit-sentiment

# 3. Work on feature (commit frequently)
git add .
git commit -m "Add Reddit PRAW integration"

# 4. Keep branch updated
git fetch origin
git rebase origin/develop

# 5. Push to remote
git push origin feature/phase3-reddit-sentiment

# 6. Create Pull Request to develop
# (via GitHub UI)

# 7. After merge, delete branch
git checkout develop
git pull origin develop
git branch -d feature/phase3-reddit-sentiment
git push origin --delete feature/phase3-reddit-sentiment
```

### Hotfix Workflow

```bash
# 1. Branch from main
git checkout main
git pull origin main
git checkout -b hotfix/1.2.1-critical-api-fix

# 2. Fix the issue
git add .
git commit -m "Fix critical API authentication issue"

# 3. Push and create PR to main
git push origin hotfix/1.2.1-critical-api-fix

# 4. After merge to main, also merge to develop
git checkout develop
git pull origin develop
git merge hotfix/1.2.1-critical-api-fix
git push origin develop

# 5. Tag the release
git checkout main
git pull origin main
git tag -a v1.2.1 -m "Hotfix: Critical API fix"
git push origin v1.2.1
```

### Release Workflow

```bash
# 1. Create release branch from develop
git checkout develop
git pull origin develop
git checkout -b release/2.0.0

# 2. Version bump and finalization
# - Update version in pyproject.toml
# - Update CHANGELOG.md
# - Final testing

git add .
git commit -m "Prepare release 2.0.0"
git push origin release/2.0.0

# 3. Create PR to main (triggers deployment)

# 4. After merge, tag the release
git checkout main
git pull origin main
git tag -a v2.0.0 -m "Release 2.0.0: Phase 3 completion"
git push origin v2.0.0

# 5. Merge back to develop
git checkout develop
git merge release/2.0.0
git push origin develop

# 6. Delete release branch
git branch -d release/2.0.0
git push origin --delete release/2.0.0
```

## 🛡️ Branch Protection Rules

### Recommended GitHub Settings

#### For `main`:
- ✅ Require pull request before merging
- ✅ Require approvals: 1
- ✅ Dismiss stale reviews on new commits
- ✅ Require status checks to pass (CI tests)
- ✅ Require branches to be up to date
- ✅ Require conversation resolution
- ✅ Restrict who can push (maintainers only)
- ✅ Do not allow force pushes
- ✅ Do not allow deletions

#### For `develop`:
- ✅ Require pull request before merging
- ✅ Require status checks to pass (CI tests)
- ✅ Do not allow force pushes
- ✅ Allow administrators to bypass

## 📝 Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `chore`: Build process, dependencies, tooling
- `ci`: CI/CD changes

### Examples:
```bash
feat(phase3): add Reddit sentiment analysis via PRAW

fix(backtest): correct equity curve calculation

docs(readme): update API key setup instructions

chore(deps): upgrade streamlit to 1.30.0
```

## 🏷️ Tagging Strategy

### Version Format: `v<major>.<minor>.<patch>`

Following [Semantic Versioning](https://semver.org/):

- **Major (v2.0.0)**: Breaking changes, major phase completion
- **Minor (v1.3.0)**: New features, backward compatible
- **Patch (v1.2.1)**: Bug fixes, no new features

### Tag Examples:
```bash
# Phase completions
v1.0.0 - Phase 1: Core Trading System
v2.0.0 - Phases 1-3: Intelligence Network complete
v3.0.0 - Phase 5: RL Agents complete

# Feature releases
v1.1.0 - Add Streamlit dashboard
v1.2.0 - Add NewsAPI integration

# Patches
v1.1.1 - Fix dashboard crash on missing data
v1.2.1 - Security hotfix for API credentials
```

### Creating Tags:
```bash
# Annotated tag (recommended)
git tag -a v1.3.0 -m "Release 1.3.0: Multi-timeframe analysis"
git push origin v1.3.0

# List tags
git tag -l

# View tag details
git show v1.3.0
```

## 🧹 Branch Cleanup

### Regular Maintenance:

```bash
# View merged branches
git branch --merged develop

# Delete local merged branches
git branch --merged develop | grep -v "^\*\|main\|develop" | xargs -n 1 git branch -d

# Prune remote-tracking branches
git fetch --prune

# View stale branches (older than 3 months)
git for-each-ref --sort=-committerdate refs/heads/ --format='%(committerdate:short) %(refname:short)'
```

### Archiving:
- Merged branches: Delete after 30 days
- Abandoned branches: Archive or delete after 90 days
- Long-running phase branches: Keep until phase completion

## 🔍 Branch Naming Best Practices

### ✅ Good Names:
- `feature/phase4-lstm-implementation`
- `feature/phase6-dashboard-charts`
- `bugfix/234-fix-sentiment-api-timeout`
- `hotfix/1.2.1-security-patch`
- `release/2.1.0`

### ❌ Bad Names:
- `my-branch` (too vague)
- `test` (not descriptive)
- `fix-stuff` (unclear what it fixes)
- `phase5` (no description)
- `johns-work` (use feature/phase format)

## 🤝 Pull Request Guidelines

### PR Title Format:
```
[Phase <N>] <Type>: <Description>
```

**Examples:**
- `[Phase 3] feat: Add Reddit sentiment analysis`
- `[Phase 2] fix: Correct Alpaca order execution`
- `[Infrastructure] chore: Upgrade Docker base image`

### PR Description Template:
```markdown
## Description
Brief description of changes

## Phase
- [ ] Phase 1: Core System
- [x] Phase 3: Intelligence Network
- [ ] Other: <specify>

## Type of Change
- [ ] New feature
- [x] Bug fix
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Manual testing completed

## Checklist
- [x] Code follows style guidelines
- [x] Self-review completed
- [x] Documentation updated
- [x] No new warnings

## Related Issues
Fixes #<issue-number>
```

## 🔗 Integration with Virtuals Protocol

For future integration with [Virtuals Protocol](https://app.virtuals.io/) (AI agent tokenization platform):

### Potential Branch Structure:
- `feature/virtuals-integration-base` - Core integration
- `feature/virtuals-agent-tokenization` - Tokenize trading agents
- `feature/virtuals-marketplace` - List agents on marketplace
- `feature/virtuals-governance` - Token-based governance

### Considerations:
- Web3 wallet integration (Phase 7+)
- Smart contract deployment (separate repo or monorepo?)
- Agent performance tracking for token holders
- Revenue sharing mechanisms

## 📊 Branch Metrics

Track these metrics monthly:
- Average branch lifetime
- Number of active branches
- Merge frequency to `develop`
- Release cadence
- Stale branches (>90 days)

## 📚 Additional Resources

- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Trunk-Based Development](https://trunkbaseddevelopment.com/)
- [Semantic Versioning](https://semver.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)

## 🆘 Need Help?

- Check documentation: `/docs/`
- Review phase guides: `/docs/phase_guides/`
- Create an issue: GitHub Issues
- Contact maintainers

---

**Remember**: Clear branch management = easier collaboration + faster development! 🚀
