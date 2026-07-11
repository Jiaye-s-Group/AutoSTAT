# Architecture

AutoSTAT has three layers.

## `core/`

Shared infrastructure that does not depend on Streamlit:

- `llm_client.py`: OpenAI-compatible chat client and JSON parsing helpers.
- `llm_providers.py`: provider catalog used by the UI and defaults.
- `config_store.py`: user-level configuration persistence under
  `~/.config/autostat`.
- `prompt_template.py`: small prompt-template renderer.
- `rag_retriever.py` and reference parsers: optional reference-document support.
- `modeling_table_utils.py`: formatting helpers for model-result tables.

`core/` should stay UI-free. If a helper needs `st.session_state`, it belongs in
`frontend/`, not here.

## `knowledge/`

The built-in statistical-method catalog used by `core/rag_retriever.py`.
Runtime code should treat it as read-only project data.

## `workflows/`

The workflow layer runs AutoSTAT's analysis stages:

- planning
- loading
- preprocessing
- visualization
- modeling
- table-of-contents generation
- report writing

Workflow functions should accept plain Python values and return dictionaries
with stable keys. This makes them reusable from the UI and runnable from
CLI entry points.

## `frontend/`

The Streamlit layer owns:

- page layout
- sidebar configuration
- session-state containers
- progress display
- report preview and download controls

The frontend should call `workflows/` through `frontend/utils/local_workflow_bridge.py`
and avoid embedding analysis logic directly in page renderers.

### Settings

`frontend/settings/llm_config.py` owns the sidebar model configuration UI. It
uses `core.llm_providers` for presets and `core.config_store` for optional
local persistence. Runtime credentials are synchronized to environment
variables for workflow workers, but are never written into the repository.

### Visualization

The visualization page has two paths:

- LLM-assisted recommendation and code generation through `workflows/`.
- Deterministic quick actions in `viz_quick_action.py` for common charts that
  should work even before calling an LLM.

### Report UI Modules

`frontend/workflow/report/report_render.py` is the Streamlit page coordinator.
It keeps page layout, report-generation actions, HTML cleanup, figure injection,
and modeling-table placement together because those pieces share the final
rendered document.

Supporting modules keep the workflow boundaries explicit:

- `report_inputs.py`: builds table-of-contents and report-writing payloads.
- `report_preview.py`: renders current or pending report previews.
- `report_export.py`: prepares HTML, Markdown, Word, and PDF downloads.
- `report_generation.py`: manages background report worker processes, polling,
  progress display, and temporary job cleanup.
- `report_state.py`: owns report session-state keys, cached outputs, and
  generation status transitions.
- `report_constants.py`: shared report field names, cache keys, and export
  defaults.

## Open-Source Boundary

This edition intentionally excludes private deployment, invite-code, quota,
WeChat, and production operations modules. Those concerns can live in a
separate downstream product repository. The open-source project should remain a
clean local-first analysis system.

## Refactoring Priorities

1. Introduce typed result schemas for workflow outputs.
2. Keep provider configuration OpenAI-compatible by default, with optional
   adapters for non-compatible APIs.
