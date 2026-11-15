# ThreatLens AI — Documentation

## Overview

ThreatLens AI is an agentic cybersecurity investigation platform that performs multi-step investigations into suspicious emails, URLs, IP addresses, and network activities. It combines real-world datasets, live threat intelligence sources, and a Python-based tool-calling agent powered by Tiny Aya Global.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your TINY_AYA_API_KEY

# Initialize database
python -c "from database.db import init_db; init_db()"

# Run the Streamlit dashboard
streamlit run app.py
```

## Project Structure

| Directory/File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard entry point |
| `agents/` | Investigation agent, planner, response generator |
| `tools/` | Classification and analysis tools (email, URL, IP, network) |
| `integrations/threat_intelligence/` | External API clients (URLhaus, PhishTank, AbuseIPDB, VirusTotal) |
| `database/` | SQLite schema and query layer |
| `datasets/` | Dataset manifest and download scripts |
| `evaluation/` | Metrics computation and benchmark runner |
| `reports/` | PDF/HTML incident report generation |
| `tests/` | Unit tests |

## Running Tests

```bash
python -m pytest tests/ -v
# Or directly:
python tests/test_tools.py
```

## Evaluation

To run the agent evaluation benchmark:

1. Ensure your Tiny Aya API key is set in `.env`
2. Open the **Evaluation** tab in the Streamlit dashboard
3. Select agent version and click "Run Benchmark"

Evaluation metrics tracked:
- Tool selection accuracy
- Argument accuracy
- Task completion rate
- Classification performance
- Evidence grounding rate
- Tool-call efficiency
- Escalation accuracy

## Datasets

Datasets are listed in [`datasets/dataset_manifest.json`](../datasets/dataset_manifest.json). To download:

```python
from datasets.download import download_phishing_email_dataset, download_phishing_detection_v2
download_phishing_email_dataset()
download_phishing_detection_v2()
```

## Configuration

All credentials go in `.env`:

| Variable | Description |
|---|---|
| `TINY_AYA_API_KEY` | Required for the agent |
| `TINY_AYA_MODEL` | LLM model name (default: tiny-aya-global) |
| `URLHAUS_API_KEY` | Optional — URLhaus API key |
| `PHishtank_API_KEY` | Optional — PhishTank API key |
| `ABUSEIPDB_API_KEY` | Optional — AbuseIPDB API key |
| `VIRUSTOTAL_API_KEY` | Optional — VirusTotal API key |

Tools gracefully handle missing API keys by skipping unavailable services.

## Security Notes

- Never auto-visits URLs on the user's device
- Treats all external data as untrusted
- Credentials never logged
- Tool arguments validated before execution
- Human approval required for any action that modifies security systems

## License

Project-specific. Datasets retain their original licenses (see [dataset_manifest.json](../datasets/dataset_manifest.json)).
