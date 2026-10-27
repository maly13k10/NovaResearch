# NovaResearch AI Premium — FINAL V5

NovaResearch AI is a research decision-support platform that turns a project PDF into a structured, reproducible workflow: project understanding → research idea → data → novelty → experiment → ML → Differential Privacy → robustness/XAI/statistics → publication package.

## Fastest local start on Windows
1. Extract the ZIP.
2. Double-click `start_nova.bat`.
3. The first run creates `.venv` and installs dependencies.
4. Open the local URL shown by Streamlit (normally `http://localhost:8501`).

## Build the Windows `.exe` + Desktop shortcut
Double-click `build_windows.bat` on Windows.

It will:
- install build dependencies;
- build `dist\NovaResearchAI.exe` with the Nova icon;
- create `NovaResearch AI.lnk` on the Desktop;
- create `professor_package\NovaResearchAI_Professor_Windows.zip`.

Alternative: push the project to GitHub and run the included **Build Windows EXE** GitHub Actions workflow. The workflow builds the Windows executable on `windows-latest` and exposes it as a downloadable artifact.

## Access from anywhere
Deploy `Nova.py` from a GitHub repository to Streamlit Community Cloud, or use the included Dockerfile on a Docker-compatible host. After deployment you receive a public HTTPS URL that works from home, university computers, phones and tablets (subject to the institution/network allowing normal HTTPS access).

See `DEPLOY_ONLINE.md`.

## Optional cloud AI for Ask Nova
Ask Nova works without an API key using the local contextual engine. For the optional OpenAI Responses API mode, configure `OPENAI_API_KEY` in Streamlit Secrets or as an environment variable. You can override the model with `NOVA_OPENAI_MODEL`.

## Main modules
- Premium SaaS UI and Research Command Center
- Competition/Jury Dashboard
- Project AI and Paper DNA
- Research Ideas + Comparison Arena
- Data Intelligence
- Novelty Radar + Heatmap + Gap Matrix
- Experiment Planner
- Synthetic robotics dataset generator
- ML benchmark
- Differential Privacy + Privacy Advisor
- Robustness + missingness + ablation + drift
- Permutation importance + real SHAP global/local
- Exact McNemar paired validation
- Research Graph
- Journal Matcher
- Context-aware Ask Nova with conversation history and PDF evidence retrieval
- Research Story Generator
- Publication Studio V2
- Reproducibility Passport
- Full DOCX + publication ZIP
- Full `.novazip` project save/load
- Live Analysis Timeline
- Professor/Jury Mode

## Scientific integrity
NovaResearch AI is a decision-support research prototype. Automated readiness/novelty/journal-fit scores are indicators, not publication guarantees. Verify literature, citations, current venue information, statistical assumptions and privacy claims before publication.
