# Development Guide

This guide explains how to set up and use the automated testing infrastructure for the cbandits project.

## Quick Start

### 1. Install Development Dependencies

```bash
# Install the package with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 2. Run Tests Locally

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run all pre-commit checks
make pre-commit
```

## Automated Testing Setup

### GitHub Actions Workflows

The project includes several GitHub Actions workflows:

1. **`.github/workflows/ci.yml`** - Main CI pipeline
   - Runs on: push to main/develop, pull requests
   - Tests on Python 3.8-3.12
   - Includes: tests, linting, type checking, examples
   - Builds package on main branch

2. **`.github/workflows/pr-checks.yml`** - Pull request specific checks
   - Security scanning with safety
   - Performance testing
   - Documentation validation

### Pre-commit Hooks

Pre-commit hooks run automatically before each commit:

- **Code formatting**: Black and isort
- **Linting**: flake8
- **Type checking**: mypy
- **General checks**: trailing whitespace, file endings, etc.

### Makefile Commands

Use the Makefile for common development tasks:

```bash
# Show all available commands
make help

# Install development dependencies
make install-dev

# Run tests
make test

# Run tests with coverage
make test-cov

# Format code
make format

# Check formatting
make format-check

# Run linting
make lint

# Check examples work
make check-examples

# Test imports
make check-imports

# Run all pre-commit checks
make pre-commit

# Clean build artifacts
make clean

# Build package
make build

# Complete development pipeline
make all
```

## Branch Protection Setup

To enable branch protection on GitHub:

1. Go to your repository settings
2. Navigate to "Branches" → "Add rule"
3. Set branch name pattern: `main`
4. Enable the following:
   - ✅ Require a pull request before merging
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Include administrators
5. Select required status checks:
   - `test / test (3.8)`
   - `test / test (3.9)`
   - `test / test (3.10)`
   - `test / test (3.11)`
   - `test / test (3.12)`
   - `lint / lint`
   - `examples / examples`
   - `security / security`
   - `performance / performance`
   - `documentation / documentation`

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes and Test Locally

```bash
# Run pre-commit checks
make pre-commit

# Or run individual checks
make test
make lint
make check-examples
```

### 3. Commit Changes

```bash
git add .
git commit -m "Add your feature description"
```

Pre-commit hooks will run automatically and fix formatting issues.

### 4. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

### 5. CI/CD Pipeline

When you create a pull request:

1. **Automated tests** run on multiple Python versions
2. **Code quality checks** ensure formatting and linting
3. **Security scanning** checks for vulnerable dependencies
4. **Performance tests** verify no regressions
5. **Documentation checks** ensure examples work

### 6. Merge to Main

Once all checks pass and the PR is approved:
- Merge the pull request
- The main branch will automatically build and test the package

## Testing Strategy

### Unit Tests

- **Location**: `tests/`
- **Coverage**: Aim for >90% code coverage
- **Run**: `pytest tests/ -v`

### Integration Tests

- **Examples**: All simple examples must run successfully
- **Imports**: All public APIs must be importable
- **Performance**: Basic performance regression testing

### Code Quality

- **Formatting**: Black (88 character line length)
- **Import sorting**: isort with Black profile
- **Linting**: flake8 with specific ignore rules
- **Type checking**: mypy with ignore-missing-imports

## Troubleshooting

### Pre-commit Hooks Fail

```bash
# Skip hooks for this commit (not recommended)
git commit --no-verify

# Run hooks manually
pre-commit run --all-files
```

### Tests Fail Locally but Pass in CI

1. Check Python version compatibility
2. Ensure all dependencies are installed: `pip install -e ".[dev]"`
3. Clear cache: `make clean`

### Import Errors

```bash
# Test imports
make check-imports

# Reinstall package
pip install -e .
```

### Performance Issues

```bash
# Run performance test locally
python -c "
import time
from cbandits import UCB_B1, GeneralCostRewardEnvironment
# ... (see pr-checks.yml for full test)
"
```

## Continuous Integration Details

### Test Matrix

- **Python versions**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Operating systems**: Ubuntu latest
- **Dependencies**: Cached for faster builds

### Coverage Reporting

- Coverage reports are generated and uploaded to Codecov
- Coverage data is available in the CI logs
- HTML coverage reports are generated locally with `make test-cov`

### Security Scanning

- **Safety**: Checks for known vulnerabilities in dependencies
- **Dependabot**: Automated dependency updates (enable in GitHub settings)

## Best Practices

1. **Always run tests locally** before pushing
2. **Use meaningful commit messages** following conventional commits
3. **Keep PRs small and focused** for easier review
4. **Add tests for new features** to maintain coverage
5. **Update documentation** when changing APIs
6. **Check performance** for algorithm changes

## Monitoring

- **GitHub Actions**: Monitor workflow runs in the Actions tab
- **Codecov**: Track test coverage trends
- **Dependabot**: Monitor for security updates
- **Performance**: Watch for performance regressions in PR checks 