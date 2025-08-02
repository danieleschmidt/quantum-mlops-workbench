# Architecture Decision Records (ADRs)

This directory contains the Architecture Decision Records for the Quantum MLOps Workbench project.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences.

## Format

Each ADR follows this format:
- **Status**: Proposed, Accepted, Superseded, Deprecated
- **Context**: What is the issue that we're seeing that is motivating this decision?
- **Decision**: What is the change that we're proposing or have agreed to implement?
- **Consequences**: What becomes easier or more difficult to do because of this change?

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [001](001-quantum-backend-abstraction.md) | Quantum Backend Abstraction Layer | Accepted | 2025-01-15 |
| [002](002-mlops-integration-strategy.md) | MLOps Integration Strategy | Accepted | 2025-01-15 |
| [003](003-testing-framework-design.md) | Quantum Testing Framework Design | Accepted | 2025-01-15 |

## Creating New ADRs

1. Copy the template: `cp template.md XXX-title.md`
2. Fill in the sections
3. Submit a PR for review
4. Update this index once merged

## Template

See [template.md](template.md) for the ADR template.