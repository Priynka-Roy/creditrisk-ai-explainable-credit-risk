# Streamlit deployment preparation

The initial delivery is local and GitHub-published. **No public application URL is claimed.** The owner will deploy after review.

## Local environment

Use Python 3.12 and a virtual environment. From this repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
streamlit run streamlit_app.py --server.port 8503
```

[Local app](http://localhost:8503) works only while the server runs on the reader's own machine. It is not a shared public URL. Use the entry point `streamlit_app.py`.

## Community Cloud

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with repository access.
2. Select `Priynka-Roy/creditrisk-ai-explainable-credit-risk`, branch `main`, entry point `streamlit_app.py`.
3. Choose Python 3.12 in advanced settings. No API keys or secrets are needed.
4. Deploy, inspect logs, and exercise all eight sections, chart filters, thresholds and stress controls.
5. Verify the resulting HTTPS URL publicly before adding a **Live app** link to README or portfolio.

The cloud platform chooses its own port; `.streamlit/config.toml` must not contain a local port. Portable paths resolve from repository files. Committed, integrity-checked precomputed artifacts supply the app: startup performs no original-data download and no training. The public dashboard reads aggregate JSON artifacts rather than deserializing untrusted estimators.

Runtime dependencies are pinned in `requirements.txt`; artifact metadata records fitted-library versions. Dependency changes require compatibility testing and potentially regenerated artifacts. Raw data and the local virtual environment are excluded from Git. CI uses deterministic fixtures and supplied lightweight artifacts rather than downloading the source workbook.

For reproduction and verification commands, see the [README](../README.md). Public deployment remains an owner action after review; it is not required for this implementation's initial delivery.

## Version 2 readiness

The entry point and eight sections remain unchanged. Both V1 and V2 aggregate artifacts are shipped and version/integrity checked. Paired inference, calibration uncertainty and development refits are precomputed offline. Interactive simulations are deterministic, cached and bounded; no model fitting, raw download or customer input is added. After reviewing V2, deploy the same repository/entry point on Python 3.12 and verify the resulting public URL before publishing it.
