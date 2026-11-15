
---
title: ThreatLens AI
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# ThreatLens AI

**Agentic Cybersecurity Investigation & Threat Intelligence Platform**

ThreatLens AI investigates suspicious emails, URLs, IP addresses and network activities using AI-powered analysis, threat intelligence and evidence-based risk assessment.

## Features

- Multi-tool cybersecurity investigation agent
- Tiny Aya integration through Hugging Face Router (OpenAI-compatible)
- Email and URL threat classification (trained scikit-learn models)
- IP and network analysis
- Threat intelligence integrations (URLhaus, OpenPhish, AbuseIPDB, VirusTotal)
- Evidence collection and correlation
- Risk assessment and escalation
- Human review and incident reporting
- PDF report generation
- AgentTune fine-tuning pipeline
- AuditKIT evaluation framework with paired baseline vs improved comparison

## Technology Stack

- Python
- Streamlit
- Tiny Aya (via Hugging Face Inference Providers / Router)
- OpenAI-compatible Chat Completions API
- Scikit-learn
- SQLite
- Hugging Face
- AgentTune (fine-tuning)
- AuditKIT (evaluation)

## Model Configuration

ThreatLens AI uses the Hugging Face Router (OpenAI-compatible) to access
`CohereLabs/tiny-aya-global:cohere` through Hugging Face Inference Providers.

Configure the following in your Hugging Face Space **Secrets**:

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | Your Hugging Face access token (required) |

Configure the following as **Variables** (optional, defaults shown):

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_MODEL_ID` | `CohereLabs/tiny-aya-global:cohere` | Model ID served via HF Router |
| `HF_PROVIDER` | `cohere` | Inference provider |
| `HF_BASE_URL` | `https://router.huggingface.co/v1` | HF Router OpenAI-compatible endpoint |
| `HF_TEMPERATURE` | `0.2` | Sampling temperature |
| `HF_MAX_TOKENS` | `2048` | Maximum tokens per response |

## Threat Intelligence Sources

| Source | Type | API Key Required | Notes |
|--------|------|-----------------|-------|
| URLhaus | URL | Optional | abuse.ch URLhaus API |
| OpenPhish | URL/Domain | No | Community Feed (https://openphish.com/feed.txt) |
| AbuseIPDB | IP | Optional | abuseipdb.com API |
| VirusTotal | URL/Domain/IP | Optional | virustotal.com API |

OpenPhish replaces PhishTank. The Community Feed requires no API key but is
subject to OpenPhish usage restrictions (non-commercial use, no redistribution).

## JSON-Action Tool Execution

Tiny Aya Global does **not** support native tool calling by default.
Instead, the investigation agent uses an explicit **JSON-action loop**:

1. The system prompt instructs the model to return exactly one JSON action per response.
2. Tool requests use `{"action": "call_tool", "tool_name": "...", "arguments": {...}, "reason": "..."}`.
3. Final conclusions use `{"action": "final", "conclusion": {...}}`.
4. The agent parses JSON (including Markdown-wrapped), validates tool names and arguments
   against the tool registry, executes only through `execute_tool()`, and appends results
   as normal conversation messages.

Only registered tools may execute. Model-generated code or arbitrary shell commands are
never executed.

## Required Workflow

```
Tiny Aya Baseline → Trajectory Collection → AgentTune Fine-Tuning →
Trained Agent → AuditKIT Evaluation → Comparative Results → Human Review
```

The investigation workflow retains human review and incident reporting as its final stage.

## AgentTune — Fine-Tuning

### Implementation status: Implemented

AgentTune fine-tuning is implemented in `evaluation/agenttune/train_agent.py` using the
supported `agenttune.train_agentic` API.

**Components:**
- `evaluation/agenttune/trajectory_collector.py` — Tiny Aya investigation trajectory collection
- `evaluation/agenttune/trajectory_validator.py` — Trajectory validation and preprocessing
- `evaluation/agenttune/agenttune_rewards.py` — Reward functions for tool selection, argument
  correctness, evidence grounding, task completion, classification accuracy, escalation accuracy,
  no duplicate calls, and no unsupported conclusions
- `evaluation/agenttune/train_agent.py` — Actual AgentTune training via `agenttune.train_agentic`
- `evaluation/agenttune/training_config.yaml` — Reproducible training configuration

**Reward functions:**
- `tool_selection` — correct tool selection (intersection / expected)
- `argument_accuracy` — correct tool arguments
- `evidence_grounding` — conclusions grounded in confirmed evidence
- `task_completion` — investigation completed/escalated
- `classification_accuracy` — correct risk classification (partial credit for adjacent levels)
- `escalation_accuracy` — correct human escalation decision
- `no_duplicate_calls` — no repeated identical tool calls
- `no_unsupported_conclusions` — High/Critical requires confirmed evidence

**Training configuration** (`evaluation/agenttune/training_config.yaml`):
- Algorithm: GRPO (supports grpo, dpo, ppo, rloo, bco)
- Base model: `CohereLabs/tiny-aya-global:cohere`
- Hardware: CPU or GPU (auto-detected; `device: auto`, `mixed_precision: fp32` on CPU)
- LoRA: rank 16, alpha 32, target modules q_proj/v_proj/k_proj/o_proj
- Reproducible: fixed seed (42), config saved to `artifacts/agenttune/checkpoints/training_config.yaml`

**Run training:**
```bash
python -m evaluation.agenttune.train_agent
```

Training artifacts (checkpoints + LoRA adapters) are saved under
`artifacts/agenttune/checkpoints/`. Trajectories are saved under
`artifacts/agenttune/trajectories/`.

**Status:** Training requires `agenttune` and `trl` to be installed and a compatible backend.
No checkpoints are fabricated — training only runs when the dependencies are available.

## AuditKIT — Evaluation

### Implementation status: Implemented

AuditKIT evaluation is implemented in `evaluation/auditkit/` using the supported
`auditkit.evaluate` API.

**Components:**
- `evaluation/auditkit/run_baseline.py` — Baseline Tiny Aya evaluation via `auditkit.evaluate`
- `evaluation/auditkit/run_improved.py` — Improved (AgentTune-trained) agent evaluation
- `evaluation/auditkit/custom_scorers.py` — Custom scorers for tool selection, argument accuracy,
  task completion, classification performance, evidence grounding, tool-call efficiency, and
  escalation accuracy
- `evaluation/auditkit/dataset_adapter.py` — AuditKIT-compatible dataset construction
- `evaluation/auditkit/compare_results.py` — Paired baseline vs improved comparison
- `evaluation/auditkit/export_report.py` — Raw outputs, metrics, and comparison reports (JSON/CSV/MD/HTML)

**Custom scorers:**
- `tool_selection_accuracy`
- `argument_accuracy`
- `task_completion_rate`
- `classification_performance`
- `evidence_grounding`
- `tool_call_efficiency`
- `escalation_accuracy`
- `tool_call_f1`
- `evidence_grounding_rate`
- `tool_call_accuracy`

**Run baseline evaluation:**
```bash
python -m evaluation.auditkit.run_baseline
```

**Run improved evaluation:**
```bash
python -m evaluation.auditkit.run_improved
```

**Compare baseline vs improved:**
```bash
python -m evaluation.auditkit.compare_results
```

**Important:** If no trained model exists (AgentTune fine-tuning not completed), the improved
evaluation reports `improved_model_unavailable` and falls back to the baseline agent. No false
improvement claims are made. The baseline is never run twice and labeled as improved.

## Dataset Sources and Licensing

See `data_pipeline/dataset_manifest.json` for full dataset metadata.

| Dataset | Source | Records | License |
|---------|--------|---------|---------|
| phishing-email-dataset | zefang-liu/phishing-email-dataset | ~18,650 | MIT |
| PhishingEmailDetection-v2 | cybersectony/PhishingEmailDetectionv2.0 | ~22,644 email / ~177,356 URL | CC-BY-SA-4.0 |
| CIC-IDS2017 | Canadian Institute for Cybersecurity | N/A (manual download) | Custom — academic use |

**CIC-IDS2017 status:** Missing. The CIC-IDS2017 network intrusion dataset is NOT bundled
with this repository. It must be downloaded manually from
https://www.unb.ca/cic/datasets/ids-2017.html and is not redistributed due to licensing.
Network analysis scenarios reference CIC-IDS2017 but the actual CSV flow files are absent.

**Evaluation scenarios:** `datasets/evaluation/scenarios.json` contains 8 held-out scenarios
(phishing email, suspicious URL, legitimate domain, network brute force, multi-indicator
phishing, clean URL, suspicious domain with no intel match, spear phishing). Each scenario has
ground-truth references for expected tools, risk level, escalation decision, and minimum evidence.

## Deployment Configuration

- **Streamlit application:** `app.py` (10 tabs: Overview, Investigate, Email, URL Scanner,
  IP Intelligence, Network Analyzer, Threat Intelligence, Execution Logs, Evaluation,
  Review & Reports)
- **PDF reporting:** `reports/report_generator.py` (WeasyPrint-based PDF generation;
  falls back to HTML if WeasyPrint system libraries are unavailable)
- **Docker:** `Dockerfile` (Python 3.12 slim, WeasyPrint system dependencies included)
- **Model artifacts:** `models/email_classifier.joblib`, `models/url_classifier.joblib`
  (trained scikit-learn pipelines; verified to load with real test accuracy metrics)
- **No Anthropic or Claude API dependencies** are used or introduced.

## Reproducibility Instructions

1. Clone the repository and checkout branch `hackathon-submission-upgrade`.
2. Install dependencies: `pip install -r requirements.txt`
3. Set `HF_TOKEN` in your environment or Hugging Face Space Secrets.
4. Run baseline evaluation: `python -m evaluation.auditkit.run_baseline`
5. Collect trajectories: `python -m evaluation.agenttune.train_agent` (collects trajectories)
6. Run AgentTune training: `python -m evaluation.agenttune.train_agent` (trains when deps available)
7. Run improved evaluation: `python -m evaluation.auditkit.run_improved`
8. Compare results: `python -m evaluation.auditkit.compare_results`
9. Run tests: `pytest tests/`

## Known Limitations

| Area | Limitation | Status |
|------|-----------|--------|
| AgentTune training | Requires `agenttune` + `trl` + compatible backend (TRL/Unsloth). CPU training is slow; GPU recommended. | Implemented, not yet executed |
| CIC-IDS2017 data | Not bundled; must be downloaded manually. | Missing (documented) |
| HF_TOKEN | Required for Tiny Aya via HF Router; absent token → graceful degradation. | Required secret |
| HF Spaces storage | Ephemeral unless persistent storage configured. | Deployment note |
| Tiny Aya tool calling | No native tool calling; uses JSON-action loop workaround. | Implemented |
| Improved model | Falls back to baseline with `improved_model_unavailable` if no trained adapter exists. | Implemented |

## Implemented vs Tested vs Incomplete

**Implemented & Tested:**
- Investigation agent (JSON-action loop, tool validation, step counting, MAX_STEPS, duplicate-call prevention)
- Evidence-sufficient termination and final conclusion processing
- Threat-intel error handling (never declares safe, conservative escalation)
- Planner domain extraction fix
- Tool argument validation (JSON Schema)
- Evidence provenance
- Model artifacts load (email + URL classifiers with real metrics)
- All 101 tests pass

**Implemented:**
- AgentTune training pipeline (genuine `agenttune.train_agentic` integration)
- AuditKIT evaluation (genuine `auditkit.evaluate` integration)
- Custom scorers (bugs fixed: argument accuracy, evidence grounding, escalation field)
- Paired baseline vs improved comparison
- Raw outputs, metrics, comparison reports

**Incomplete / Not yet executed:**
- AgentTune fine-tuning has not been executed (requires GPU/time; no fabricated checkpoints)
- Improved model evaluation reports `improved_model_unavailable` until training is run
- CIC-IDS2017 dataset is missing

## Important

This is a cybersecurity research and demonstration application. Investigation results require
human review and should not be treated as definitive security guarantees.

AgentTune fine-tuning and AuditKIT comparative experiments are implemented but training has not
been executed in this submission. No checkpoints, metrics, or deployment claims are fabricated.