# NovaResearch AI V5 — Validation Report

## Completed in the build environment
- Python syntax compilation for `Nova.py`, launcher and all `core/` modules: PASS
- Core smoke test: PASS
  - domain detection / readiness
  - synthetic robotics dataset
  - ML benchmark
  - Differential Privacy engine or labelled fallback
  - permutation importance
  - noise robustness
  - missing-data robustness
  - automatic ablation
  - sensor drift
  - exact McNemar p-value
  - session ZIP save/load
  - DOCX Paper Builder
  - complete Publication ZIP builder

Smoke test output: `NOVA_SMOKE_TEST_OK`

## Environment limitation
The sandbox used to assemble the project did not have Streamlit preinstalled and did not allow package downloads from PyPI, so a live browser launch of Streamlit could not be performed here. The complete source passes Python compilation and the application-independent engines pass the smoke test. `start_nova.bat` installs Streamlit/dependencies on the user's internet-connected Windows PC before launch.
