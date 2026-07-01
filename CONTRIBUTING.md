# Collaboration conventions

This repository is shared by people and multiple coding agents.

## Shared assets

Source code, scripts, tests, generated experiment outputs, figures, and
method documentation are repository-wide assets. Name them for the method or
purpose, not for the person or agent that created them. Anyone may use,
review, or extend these assets.

Current example: `scripts/pdk_mlp_tandem.py` and `out/pdk_mlp_tandem_*` are
shared MLP experiment assets.

## Agent work logs

Keep agent-authored Markdown working notes separate so one agent does not
overwrite another agent's context or history.

- Codex notes: `docs/CODEX_RESEARCH_LOG.md`
- Claude notes: leave existing Claude-authored Markdown files unchanged
- Shared, verified conclusions may be promoted into neutral project
  documentation after review

Do not use agent-specific names for executable code or experiment output
directories.
