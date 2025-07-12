# Automated Testing Setup Guide

This guide will help you set up automated testing to ensure code quality before merging to main.

## 🚀 Quick Setup (5 minutes)

### 1. Install Development Tools

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 2. Test Your Setup

```bash
# Run all checks locally
make pre-commit

# Or test individual components
make test
make lint
make check-examples
```

## 📋 What's Been Set Up

### ✅ GitHub Actions Workflows

1. **Main CI Pipeline** (`.github/workflows/ci.yml`)
   - Runs on: push to main/develop, pull requests
   - Tests on Python 3.8-3.12
   - Includes: tests, linting, type checking, examples
   - Builds package on main branch

2. **Pull Request Checks** (`.github/workflows/pr-checks.yml`)
   - Security scanning with safety
   - Performance testing
   - Documentation validation

### ✅ Pre-commit Hooks

Automatically run before each commit:
- Code formatting (Black, isort)
- Linting (flake8)
- Type checking (mypy)
- General file checks

### ✅ Makefile Commands

Quick commands for development:
```bash
make help          # Show all commands
make test          # Run tests
make lint          # Run linting
make format        # Format code
make pre-commit    # Run all checks
```

## 🔒 Enable Branch Protection

**Important**: You must manually enable branch protection on GitHub:

1. Go to your repository on GitHub
2. Settings → Branches → Add rule
3. Branch name pattern: `main`
4. Enable these options:
   - ✅ Require a pull request before merging
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Include administrators

5. Select these required status checks:
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

## 🔄 Development Workflow

### Before Making Changes

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Ensure you have latest changes
git pull origin main
```

### While Developing

```bash
# Run checks frequently
make pre-commit

# Or run specific checks
make test
make lint
make check-examples
```

### Before Committing

```bash
# Pre-commit hooks run automatically
git add .
git commit -m "Add your feature"

# If hooks fail, fix issues and commit again
```

### Before Pushing

```bash
# Final check before pushing
make pre-commit

# Push your branch
git push origin feature/your-feature-name
```

### Create Pull Request

1. Go to GitHub and create a pull request
2. All CI checks will run automatically
3. Wait for all checks to pass (green checkmarks)
4. Request review if needed
5. Merge only when all checks pass

## 🛠️ Troubleshooting

### Pre-commit Hooks Fail

```bash
# Run hooks manually to see detailed errors
pre-commit run --all-files

# Skip hooks for this commit (not recommended)
git commit --no-verify
```

### Tests Fail Locally

```bash
# Clean and reinstall
make clean
pip install -e ".[dev]"

# Run tests with verbose output
pytest tests/ -v -s
```

### Import Errors

```bash
# Test imports
make check-imports

# Reinstall package
pip install -e .
```

### CI Fails but Local Works

1. Check Python version compatibility
2. Ensure all dependencies are in `pyproject.toml`
3. Check for platform-specific issues
4. Look at CI logs for specific error messages

## 📊 Monitoring

### GitHub Actions
- Monitor workflow runs in the Actions tab
- Check for failed jobs and fix issues
- Review performance trends

### Code Coverage
- Coverage reports are generated in CI
- Run locally with: `make test-cov`
- Aim for >90% coverage

### Security
- Safety checks run on every PR
- Monitor for dependency vulnerabilities
- Enable Dependabot for automated updates

## 🎯 Best Practices

1. **Always run tests locally** before pushing
2. **Keep PRs small and focused** for easier review
3. **Add tests for new features** to maintain coverage
4. **Use meaningful commit messages**
5. **Check performance** for algorithm changes
6. **Update documentation** when changing APIs

## 📞 Getting Help

- Check the full [Development Guide](DEVELOPMENT.md) for detailed information
- Look at CI logs for specific error messages
- Review the [README](README.md) for project overview
- Check [installation.md](installation.md) for installation issues

## ✅ Verification Checklist

Before merging to main, ensure:

- [ ] All tests pass locally (`make test`)
- [ ] Code is properly formatted (`make format-check`)
- [ ] Linting passes (`make lint`)
- [ ] Examples work (`make check-examples`)
- [ ] All CI checks pass on GitHub
- [ ] Code review is complete
- [ ] Documentation is updated
- [ ] No security issues detected

---

**That's it!** Your automated testing is now set up. Every pull request will be automatically tested before it can be merged to main. 