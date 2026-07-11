# Contributing

Thank you for improving AutoSTAT.

## Local Checks

```bash
pip install -e ".[dev]"
ruff check .
```

## Code Style

- Keep workflow code independent from Streamlit.
- Prefer small pure helpers over UI-side data manipulation.
- Keep comments useful and brief. Explain why a branch exists, not what a simple
  assignment does.

## LLM-Dependent Changes

Avoid changes that require live API calls during import or app startup. Keep LLM
calls inside workflow execution paths.
