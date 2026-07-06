# Contributing to TriMixGen

Thank you for your interest in contributing to TriMixGen! As this is a research-oriented project, we adhere to professional open-source standards.

### Git Branching Strategy
All work must happen on feature branches. Do not commit directly to `main`.
- Feature branches: `feature/<module>-<description>` (e.g., `feature/data-cleaning`)
- Bug fixes: `bugfix/<description>`
- Documentation: `docs/<description>`

### Commit Message Convention
We use [Conventional Commits](https://www.conventionalcommits.org/).
- `feat(data): add unicode normalization`
- `fix(model): resolve CUDA memory leak in generation`
- `docs: update dataset links`
- `chore: update requirements.txt`

### Pull Request Process
1. Create a feature branch.
2. Ensure pre-commit hooks (`black`, `ruff`) pass locally.
3. Submit a PR against `main`.
4. A PR must describe the **Why** and **How** of the change. Include experiment metrics (like loss curves) if submitting a model update.
