# Streamlit Retirement Checklist

Tracks all changes needed to fully retire the Streamlit dashboard once React frontend reaches feature parity.

## Streamlit Files to Archive/Delete

### Dashboard entry point
- `vibe_quant/dashboard/app.py`

### Pages (7 files)
- `vibe_quant/dashboard/pages/__init__.py`
- `vibe_quant/dashboard/pages/data_management.py`
- `vibe_quant/dashboard/pages/strategy_management.py`
- `vibe_quant/dashboard/pages/discovery.py`
- `vibe_quant/dashboard/pages/backtest_launch.py`
- `vibe_quant/dashboard/pages/results_analysis.py`
- `vibe_quant/dashboard/pages/paper_trading.py`
- `vibe_quant/dashboard/pages/settings.py`

### Components (14 files)
- `vibe_quant/dashboard/components/__init__.py`
- `vibe_quant/dashboard/components/backtest_config.py`
- `vibe_quant/dashboard/components/condition_builder.py`
- `vibe_quant/dashboard/components/form_state.py`
- `vibe_quant/dashboard/components/indicator_catalog.py`
- `vibe_quant/dashboard/components/job_status.py`
- `vibe_quant/dashboard/components/overfitting_panel.py`
- `vibe_quant/dashboard/components/preflight_summary.py`
- `vibe_quant/dashboard/components/risk_management.py`
- `vibe_quant/dashboard/components/strategy_card.py`
- `vibe_quant/dashboard/components/strategy_wizard.py`
- `vibe_quant/dashboard/components/sweep_builder.py`
- `vibe_quant/dashboard/components/template_selector.py`
- `vibe_quant/dashboard/components/time_filters.py`
- `vibe_quant/dashboard/components/validation_summary.py`

### Utilities
- `vibe_quant/dashboard/__init__.py`
- `vibe_quant/dashboard/utils.py`
- `vibe_quant/dashboard/charts.py`
- `vibe_quant/dashboard/data_builders.py`

### Config
- `.streamlit/config.toml`

**Total: ~27 files**

## Python Dependencies to Remove

In `pyproject.toml` under `[project] dependencies`:
- `streamlit>=1.40.0` -- remove entirely
- `plotly>=5.24.0` -- remove if only used by Streamlit charts (verify React frontend uses its own plotly.js)

## SPEC.md Sections to Update

- Section 10 (Dashboard): Update to reference React frontend, remove Streamlit mentions
- Any `streamlit run` commands in quickstart/setup sections
- Architecture diagrams referencing Streamlit

## CLAUDE.md Updates

- Remove `Dashboard:` line referencing `.venv/bin/streamlit run ...`
- Remove Streamlit navigation page references in agent-browser section
- Update test flow to reference React frontend dev server

## Other Config Changes

- Remove `.streamlit/` directory entirely
- Update any CI/CD scripts that start Streamlit
- Update `README.md` dashboard references

## Pre-Retirement Verification

Before deleting, confirm:
- [ ] All 7 Streamlit pages have React equivalents with feature parity
- [ ] E2E tests pass for all React pages
- [ ] FastAPI backend serves all data previously fetched by Streamlit directly
- [ ] No Python code outside `vibe_quant/dashboard/` imports from dashboard modules
