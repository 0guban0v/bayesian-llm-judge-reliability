# Bayesian LLM Judge Reliability

Bayesian Item Response Theory (IRT) for measuring how reliable local LLM judges are on JudgeBench. Current experiment compares multiple local models under one fixed verdict-only judging protocol and fits Bayesian 1PL or 2PL IRT models in NumPyro.

> Public viewing only. All rights reserved. No reuse, copying, modification, or
> redistribution is permitted without prior written permission.

## Quick Start

```bash
uv sync
make pre-commit-install
make setup-models
make run
```

## Docs

- [Docs Index](docs/README.md)
- [Workflow](docs/workflow.md)
- [Structure](docs/structure.md)

## License

This repository is public for viewing only. All rights are reserved.

No use, copying, modification, distribution, or derivative works are permitted
without prior written permission from the author.
