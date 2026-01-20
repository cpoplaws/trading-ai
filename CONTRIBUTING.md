# Contributing to Trading-AI

Thank you for your interest in contributing to Trading-AI! This document provides guidelines and workflows for contributing to the project.

## 🌟 Quick Start

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/trading-ai.git`
3. **Create a branch** following our naming convention (see [Branch Management](#branch-management))
4. **Make your changes** with clear, focused commits
5. **Test** your changes thoroughly
6. **Push** to your fork
7. **Create a Pull Request** to the `develop` branch

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Branch Management](#branch-management)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## 🤝 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers warmly
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards other contributors

## 🎯 How Can I Contribute?

### 1. Report Bugs

Found a bug? Please create an issue with:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error logs/screenshots if applicable

**Use label**: `bug`

### 2. Suggest Features

Have an idea? Create an issue with:
- Clear feature description
- Use case and benefits
- Possible implementation approach
- Which phase it relates to (Phase 1-12)

**Use label**: `enhancement`, `phase-<N>`

### 3. Improve Documentation

Documentation improvements are always welcome:
- Fix typos or clarify explanations
- Add examples
- Improve README or guides
- Add inline code comments where needed

**Use label**: `documentation`

### 4. Submit Code

See [Pull Request Process](#pull-request-process) below.

## 🌿 Branch Management

We follow a structured branching strategy. **See [BRANCH_MANAGEMENT.md](BRANCH_MANAGEMENT.md) for complete details.**

### Quick Reference:

| Branch Type | Format | Example |
|:------------|:-------|:--------|
| Feature | `feature/phase<N>-<description>` | `feature/phase3-reddit-sentiment` |
| Bugfix | `bugfix/<issue>-<description>` | `bugfix/123-fix-api-error` |
| Hotfix | `hotfix/<version>-<description>` | `hotfix/1.2.1-critical-fix` |
| Release | `release/<version>` | `release/2.0.0` |

**Key Rules:**
- Feature branches from `develop`
- PRs target `develop` (not `main`)
- Keep branches focused and short-lived
- Delete after merge

## 🛠️ Development Setup

### Prerequisites

- Python 3.11+
- pip package manager
- Git
- (Optional) Docker for containerized development

### Setup Steps

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/trading-ai.git
cd trading-ai

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Install pre-commit hooks
pre-commit install

# 5. Copy environment template
cp .env.example .env
# Edit .env and add your API keys (optional for development)

# 6. Run tests to verify setup
pytest tests/ -v
```

### Development Tools

We use these tools (configured in `pyproject.toml` and `.pre-commit-config.yaml`):

- **Black**: Code formatting (line length: 100)
- **isort**: Import sorting
- **Ruff**: Fast linting
- **mypy**: Static type checking
- **pytest**: Testing framework
- **pre-commit**: Automated checks

## 📝 Coding Standards

### Python Style Guide

Follow [PEP 8](https://pep8.org/) with these specifics:

- **Line length**: 100 characters (Black default)
- **Imports**: Sorted with `isort`
- **Type hints**: Required for public functions
- **Docstrings**: Required for public modules, classes, functions

### Code Quality Checklist

```python
# ✅ Good: Type hints, docstring, clear naming
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Series of price data
        period: Lookback period for RSI calculation
        
    Returns:
        Series with RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ❌ Bad: No types, no docstring, unclear naming
def calc(p, n=14):
    d = p.diff()
    g = (d.where(d > 0, 0)).rolling(window=n).mean()
    l = (-d.where(d < 0, 0)).rolling(window=n).mean()
    return 100 - (100 / (1 + g/l))
```

### Project-Specific Conventions

1. **Logging**: Use module-level logger
   ```python
   import logging
   logger = logging.getLogger(__name__)
   logger.info("Processing started")
   ```

2. **Configuration**: Use `config/settings.yaml` for settings
3. **Error Handling**: Graceful degradation with informative logs
4. **API Keys**: Always use environment variables, never hardcode

### File Organization

```
src/
├── data_ingestion/      # Data fetching and storage
├── feature_engineering/ # Technical indicators
├── modeling/            # ML models
├── strategy/            # Trading strategies
├── execution/           # Order execution
├── backtesting/         # Performance analysis
├── monitoring/          # Dashboard and UI
└── utils/               # Shared utilities
```

## 🧪 Testing Guidelines

### Test Requirements

- All new features must include tests
- Bug fixes should include regression tests
- Maintain or improve code coverage
- Tests must pass before PR approval

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_trading_ai.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only fast tests (skip integration tests)
pytest tests/ -m "not slow"
```

### Writing Tests

```python
import pytest
from src.feature_engineering.feature_generator import FeatureGenerator

class TestFeatureGenerator:
    """Test suite for FeatureGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create a FeatureGenerator instance."""
        return FeatureGenerator(['AAPL'])
    
    def test_calculate_sma(self, generator):
        """Test SMA calculation returns expected values."""
        # Arrange
        prices = pd.Series([10, 11, 12, 13, 14])
        
        # Act
        sma = generator.calculate_sma(prices, period=3)
        
        # Assert
        assert len(sma) == len(prices)
        assert sma.iloc[-1] == pytest.approx(13.0)
```

### Test Organization

```
tests/
├── unit/              # Fast, isolated unit tests
├── integration/       # Tests with external dependencies
├── test_*.py         # Legacy test files (migrate to subdirs)
└── conftest.py       # Shared fixtures
```

## 💬 Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

### Format:
```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

### Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `perf`: Performance improvement
- `test`: Testing
- `chore`: Maintenance
- `ci`: CI/CD changes

### Examples:

```bash
# Feature with scope
feat(phase3): add Reddit sentiment analysis via PRAW

# Bug fix
fix(backtest): correct equity curve calculation for short positions

# Documentation
docs(readme): update installation instructions for TA-Lib

# Breaking change
feat(api)!: redesign strategy interface for multi-timeframe support

BREAKING CHANGE: Strategy classes must now implement get_signals_multi_timeframe()
```

### Guidelines:
- Use imperative mood ("add" not "added")
- Keep subject line under 72 characters
- Separate subject from body with blank line
- Wrap body at 72 characters
- Reference issues: "Fixes #123" or "Closes #456"

## 🔄 Pull Request Process

### Before Creating PR

1. **Sync with develop**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout your-feature-branch
   git rebase develop
   ```

2. **Run quality checks**
   ```bash
   # Format code
   black src/ tests/
   isort src/ tests/
   
   # Lint
   ruff check src/ tests/
   
   # Type check
   mypy src/
   
   # Test
   pytest tests/ -v
   ```

3. **Review your changes**
   ```bash
   git diff develop..your-feature-branch
   ```

### PR Title Format

```
[Phase <N>] <type>: <description>
```

Examples:
- `[Phase 3] feat: Add Reddit sentiment analysis`
- `[Phase 2] fix: Correct Alpaca order execution logic`
- `[Infrastructure] chore: Upgrade dependencies`

### PR Description Template

```markdown
## Description
Clear summary of what this PR does and why.

## Phase
- [ ] Phase 1: Core System
- [x] Phase 3: Intelligence Network
- [ ] Infrastructure/Other

## Type of Change
- [x] New feature
- [ ] Bug fix
- [ ] Breaking change
- [ ] Documentation update

## Changes Made
- Added Reddit PRAW integration
- Implemented sentiment scoring algorithm
- Added tests for sentiment analysis
- Updated documentation

## Testing
- [x] All existing tests pass
- [x] New tests added and passing
- [x] Manual testing completed
- [ ] Integration tests added

## Screenshots (if applicable)
<Add screenshots for UI changes>

## Checklist
- [x] Code follows project style guidelines
- [x] Self-review completed
- [x] Comments added for complex logic
- [x] Documentation updated
- [x] No new warnings
- [x] Tests added/updated
- [x] Changelog updated (if applicable)

## Related Issues
Closes #123
Relates to #456

## Notes for Reviewers
Any specific areas you'd like feedback on.
```

### After Creating PR

1. **Wait for CI checks** to pass
2. **Address review comments** promptly
3. **Keep PR updated** with develop if needed
4. **Squash commits** if requested
5. **Celebrate** when merged! 🎉

### PR Size Guidelines

- **Small** (< 200 lines): Preferred, faster review
- **Medium** (200-500 lines): Acceptable
- **Large** (> 500 lines): Break into smaller PRs if possible

## 🐛 Issue Guidelines

### Before Creating Issue

- Search existing issues to avoid duplicates
- Check if it's already fixed in `develop` branch
- Gather necessary information

### Bug Report Template

```markdown
**Bug Description**
Clear, concise description of the bug.

**To Reproduce**
1. Step 1
2. Step 2
3. See error

**Expected Behavior**
What should happen.

**Actual Behavior**
What actually happens.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.11.5]
- Trading-AI version: [e.g., 1.2.0]
- Installation method: [Docker/pip/source]

**Logs/Screenshots**
Paste relevant logs or add screenshots.

**Additional Context**
Any other relevant information.
```

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature.

**Problem It Solves**
What problem does this address?

**Proposed Solution**
How would you implement this?

**Alternatives Considered**
Other approaches you've thought of.

**Related Phase**
Which project phase does this relate to? (Phase 1-12)

**Additional Context**
Mockups, examples, references.
```

## 🏷️ Labels

We use these labels to organize issues and PRs:

### Type Labels
- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `question`: Further information requested
- `help wanted`: Extra attention needed
- `good first issue`: Good for newcomers

### Phase Labels
- `phase-1`: Core Trading System
- `phase-2`: Broker Integration
- `phase-3`: Intelligence Network
- `phase-4`: Deep Learning
- `phase-5`: RL Agents
- `phase-6`: Dashboard
- `phase-7`: Infrastructure
- `phase-8-12`: Future phases

### Priority Labels
- `priority-high`: Critical issues
- `priority-medium`: Important but not urgent
- `priority-low`: Nice to have

### Status Labels
- `in-progress`: Currently being worked on
- `blocked`: Waiting on dependencies
- `needs-review`: Ready for review
- `wontfix`: Will not be addressed

## 🎓 Learning Resources

### For New Contributors

- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)
- [Python Best Practices](https://docs.python-guide.org/)
- [Trading Concepts](docs/README.md)
- [Project Phase Guides](docs/phase_guides/)

### Project-Specific Docs

- [README.md](README.md): Project overview
- [BRANCH_MANAGEMENT.md](BRANCH_MANAGEMENT.md): Branching strategy
- [QUICKSTART.md](QUICKSTART.md): Quick setup guide
- [GETTING_STARTED.md](GETTING_STARTED.md): Detailed setup
- [docs/phase_guides/](docs/phase_guides/): Phase-specific guides

## 🚀 Advanced Topics

### Adding New Data Sources

1. Create module in `src/data_ingestion/`
2. Follow error handling patterns
3. Add API key to `.env.template`
4. Update documentation
5. Add tests

### Implementing New Strategies

1. Create class in `src/strategy/`
2. Inherit from base strategy class
3. Implement required methods
4. Add backtesting
5. Document strategy logic
6. Add to advanced strategies suite

### Adding ML Models

1. Create module in `src/modeling/`
2. Follow scikit-learn patterns
3. Add model persistence
4. Include hyperparameter tuning
5. Add evaluation metrics
6. Update model comparison

## 🤔 Questions?

- **General questions**: Create a discussion on GitHub
- **Bug reports**: Create an issue
- **Security issues**: Email maintainers (see SECURITY.md)
- **Feature ideas**: Create an issue with `enhancement` label

## 🙏 Recognition

Contributors are recognized in:
- Project README
- Release notes
- GitHub contributors page

Thank you for contributing to Trading-AI! 🚀📈

---

**Remember**: Quality over quantity. A well-tested, documented small PR is better than a large unfocused one.
