# Contributing to Noita RL

Thank you for your interest in contributing! Here's how you can help.

## 🐛 Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected vs actual behavior**
- **Screenshots** if applicable
- **Environment**: OS, Python version, Noita version

## 💡 Suggesting Features

- Check if the feature is already on the [ROADMAP.md](ROADMAP.md)
- Open an issue with the `enhancement` label
- Describe the use case and why it would benefit the project

## 🔧 Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code, add tests
3. Ensure the test suite passes (`pytest`)
4. Follow the existing code style
5. Update documentation if needed
6. Submit the PR with a clear description of changes

## 🎮 Noita Modding

This project interacts with Noita via Lua mods and `pollnet.dll`. If you're contributing to the game-side code:

- Follow the [Noita Modding Agreement](docs/Noita-ModdingAgreement-v100.rtf)
- Test changes in a clean Noita installation
- Document any new Lua API usage

## 📝 Code Style

- Python: Follow [PEP 8](https://peps.python.org/pep-0008/)
- Lua: Use consistent indentation (4 spaces)
- Commit messages: Use [Conventional Commits](https://www.conventionalcommits.org/)

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.
```

---

Questions? Open an issue or reach out to the maintainer.
