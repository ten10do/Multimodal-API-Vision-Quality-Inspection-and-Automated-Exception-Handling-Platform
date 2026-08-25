# Contributing

Contributions are welcome when they preserve the repository's evidence boundaries and keep industrial claims reproducible.

## Before Opening a Change

Use an issue for behavior changes, new integrations, or changes to public contracts. Security concerns must follow [SECURITY.md](SECURITY.md), not a public issue.

Do not commit:

- credentials, tokens, connection strings, plant addresses, or proprietary production data;
- model weights, whitening artifacts, memory banks, runtime databases, or generated experiment outputs;
- vendor screenshots or factory images without documented permission;
- claims of physical deployment when the evidence is simulator-backed.

Changes to the frozen D3 model, artifact identities, threshold, image-score path, or industrial fail-closed behavior require a separately approved validation scope. A documentation or portfolio change must not modify them.

## Development Checks

Run checks relevant to the files you changed:

```powershell
# Documentation
.\.venv\Scripts\python.exe scripts\check_docs.py

# Python default suite
.\.venv\Scripts\python.exe -m pytest -q

# Frontend
Set-Location frontend
npm test
npm run build
```

Before committing, run `git diff --check` and inspect the final changed-file list. Environment-specific GPU, artifact, OPC UA, and industrial E2E gates are documented in [the test matrix](docs/test-matrix.md).

## Pull Requests

A pull request should:

- explain the problem and the smallest change that solves it;
- list affected components and explicitly identify any model, artifact, threshold, protocol, or runtime impact;
- distinguish Implemented capability, Simulation, and Future deployment claims;
- include test commands and results;
- link evidence for metric, performance, and industrial-behavior claims;
- keep unrelated refactoring and formatting out of the diff.

Documentation links must be relative and valid on a case-sensitive filesystem. Mermaid diagrams must pass the documentation workflow. New demo material must be clearly labeled as simulator-backed unless verifiable site evidence and publication permission exist.

## Commit Style

Use concise imperative subjects with a conventional prefix where practical, for example:

- `docs: clarify SAT evidence boundary`
- `fix: preserve HOLD on adapter timeout`
- `test: cover artifact identity mismatch`
- `ci: validate documentation links`

By contributing, you agree that your contribution is licensed under the repository's [Apache License 2.0](LICENSE).
