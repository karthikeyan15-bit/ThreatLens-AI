# ThreatLens AI — System Design Document

## 1. Requirements Gathering

### Functional Requirements
- **Email Threat Analysis**: Upload/paste email content → extract headers, URLs, domains → classify as phishing/legitimate/inconclusive with evidence.
- **URL Investigation**: Validate URL → extract domain/subdomain → query threat intel (URLhaus, PhishTank) → assess risk.
- **IP/Domain Intelligence**: Check IP reputation via AbuseIPDB/VirusTotal; resolve domain → check MX/DNS records.
- **Network Log Analyzer**: Parse uploaded CSV (CIC-IDS2017 format) → detect anomalous flows → classify attack type.
- **Agentic Investigation**: Multi-step loop where Tiny Aya Global selects tools, executes them, correlates results, and escalates when uncertain.
- **Threat Correlation**: Build a graph linking indicators across emails, URLs, IPs, and logs.
- **Risk Assessment Engine**: Score evidence into Low/Medium/High/Critical/Inconclusive with source attribution.
- **Execution Monitoring**: Record every tool call, argument, result, duration, and agent decision.
- **Evaluation Dashboard**: Compare baseline vs improved agent on 7 standard metrics.
- **Human Review & Reports**: Analyst approves/rejects recommendations; generates PDF incident reports.

### Non-Functional Requirements
- **Latency**: Single-tool queries <3s; full agentic investigations <60s target.
- **Availability**: Standalone local-first; works offline with cached datasets. API calls are best-effort with graceful fallback.
- **Scalability**: Designed for a single analyst workstation; no horizontal scaling needed initially.
- **Security**: Never auto-visits URLs; treats all external data as untrusted; credentials in env vars only.
- **Reproducibility**: All dataset downloads scripted; fixed random seeds; hash-verified artifacts.

### Constraints
- Python 3.10+ / Streamlit frontend
- Tiny Aya Global as primary LLM via Cohere V2 API
- SQLite for persistence (no external DB dependency)
- No MCP — pure Python tool-calling loop
- Hackathon timeline: ~2 weeks to working demo

---

## 2. High-Level Design

### Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                      │
│  [Overview] [Email] [URL] [IP] [Network] [Agent] [Intel]   │
│            [Logs] [Eval] [Review & Reports]                 │
└──────────────────────────┬──────────────────────────────────┘
                           │ user input / investigation request
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Investigation Agent (Tool-Calling Loop)        │
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐  │
│  │   Planner   │──▶│   Agent     │──▶│ Response Gen.   │  │
│  │ (extract    │   │ (Tiny Aya   │   │ (format final   │  │
│  │  indicators)│   │  loops)     │   │  report)        │  │
│  └─────────────┘   └──────┬──────┘   └─────────────────┘  │
│               ┌────────────┘                                │
│               │ tool calls                                  │
│               ▼                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Tool Registry                          │   │
│  │  email_analyzer | url_analyzer | ip_analyzer        │   │
│  │  network_analyzer | correlate | risk_assess         │   │
│  └──────┬──────────┬──────────┬──────────┬────────────┘   │
└─────────┼──────────┼──────────┼──────────┼─────────────────┘
          │          │          │          │
          ▼          ▼          ▼          ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────────┐
│ Classification│ │ URL Haus   │ │ AbuseIPDB  │ │  VirusTotal      │
│ Model (sklearn│ │ API Client │ │ API Client │ │  API Client      │
│  pipeline)  │ │            │ │            │ │                  │
└─────────────┘ └─────────────┘ └─────────────┘ └──────────────────┘
                                                    │
          ┌─────────────────────────────────────────┤
          │                                         │
          ▼                                         ▼
┌─────────────────────┐             ┌──────────────────────────┐
│  SQLite Database    │             │  Hugging Face Datasets   │
│  - investigations   │◀───────────▶│  - phishing-email-dataset│
│  - tool_calls       │             │  - PhishingEmailDet.v2.0 │
│  - audit_log        │             │  - CIC-IDS2017 samples   │
│  - review_records   │             └──────────────────────────┘
└─────────────────────┘
```

### Data Flow (Agentic Investigation)

```
User submits investigation request
         │
         ▼
  Extract Indicators (emails, URLs, IPs, domains from text)
         │
         ▼
  Plan: which tools to call and in what order
         │
         ├─▶ email_analyzer(payload) ──┐
         ├─▶ url_analyzer(url)   ──┐  │
         ├─▶ ip_analyzer(ip)     ──┤──┼──▶ Collect all evidence
         ├─▶ network_analyzer(csv)│  │      │
         └─▶ threat_intel_queries────────┘      │
                                                ▼
                                    Threat Correlation Engine
                                    (build indicator graph)
                                                │
                                                ▼
                                    Risk Assessment Engine
                                    (score evidence → risk level)
                                                │
                     ┌──────────────────────────┤
                     │ Evidence sufficient?      │
                     ├─ Yes ──▶ Generate Report │
                     └─ No  ──▶ Escalate to Human│
                                              │
                                              ▼
                                    Human Review & Approval
                                              │
                                        Generate PDF Report
```

### Storage Choices

| Data | Storage | Reasoning |
|------|---------|-----------|
| Investigation metadata & logs | SQLite (on disk) | Zero config, single-file, ACID, easy backup |
| Processed datasets | `datasets/processed/` (parquet/CSV) | Reproducible, version-controlled via manifest |
| Evaluation trajectories | SQLite `evaluation_runs` table | Fast querying for metric aggregation |
| Threat intel caches | In-memory + SQLite cache | Reduce repeated API calls during an investigation |

---

## 3. Deep Dive

### 3.1 Data Models

**investigations**
```sql
id              TEXT PRIMARY KEY,
created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
type            TEXT,           -- 'email'|'url'|'ip'|'network'|'multi'
input_content   TEXT,           -- original user input (sanitized)
risk_level      TEXT,           -- 'Low'|'Medium'|'High'|'Critical'|'Inconclusive'
confidence      REAL,
summary         TEXT,
status          TEXT,           -- 'pending'|'running'|'completed'|'escalated'
human_approved  BOOLEAN,
reviewer_note   TEXT
```

**tool_calls**
```sql
id              INTEGER PRIMARY KEY AUTOINCREMENT,
investigation_id TEXT REFERENCES investigations(id) ON DELETE CASCADE,
step_order      INTEGER NOT NULL,
tool_name       TEXT NOT NULL,
arguments       TEXT,           -- JSON
result          TEXT,           -- JSON or error string
duration_ms     INTEGER,
timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
```

**evidence_items**
```sql
id              INTEGER PRIMARY KEY AUTOINCREMENT,
investigation_id TEXT REFERENCES investigations(id) ON DELETE CASCADE,
indicator_type  TEXT NOT NULL CHECK(indicator_type IN ('email', 'url', 'ip', 'domain')),
indicator_value TEXT NOT NULL,
source_tool     TEXT,
source_record   TEXT,
veracity        TEXT DEFAULT 'unknown' CHECK(veracity IN ('confirmed', 'suggested', 'unknown')),
confidence      REAL
```

**correlations**
```sql
id              INTEGER PRIMARY KEY AUTOINCREMENT,
investigation_id TEXT REFERENCES investigations(id) ON DELETE CASCADE,
indicator_a     TEXT NOT NULL,
indicator_b     TEXT NOT NULL,
relationship    TEXT NOT NULL,
strength        REAL
```

**evaluation_runs**
```sql
id              INTEGER PRIMARY KEY AUTOINCREMENT,
agent_version   TEXT,           -- 'baseline'|'improved'
scenario_id     TEXT NOT NULL,
tool_selection_correct BOOLEAN,
argument_accuracy REAL,         -- 0..1
task_completed  BOOLEAN,
classification_correct BOOLEAN,
evidence_grounding REAL,        -- 0..1
tool_call_count INTEGER,
escalation_appropriate BOOLEAN,
actual_output   TEXT,           -- JSON
reference_output TEXT           -- JSON
```

### 3.2 Tool Interface

All tools conform to a common signature:

```python
@dataclass
class ToolResult:
    success: bool
    data: dict[str, Any]
    errors: list[str]
    metadata: dict[str, Any]   # source, timestamp, cache_hit, etc.
```

Each tool is registered with metadata used by the agent for tool selection:
```python
ToolDefinition(
    name: str,
    description: str,
    parameters: dict,   # JSON Schema
    category: str,      # 'classification'|'intelligence'|'network'|'analysis'
)
```

### 3.3 Agentic Loop Design

The investigation agent uses a Cohere V2 tool-calling loop:

```python
async def run_investigation(request: str) -> InvestigationResult:
    state = AgentState(investigation_id=generate_id(), history=[])
    
    while not state.done and state.steps < MAX_STEPS:
        # LLM decides next action
        response = await llm.complete(
            messages=state.history,
            tools=TOOL_REGISTRY,
            max_tokens=1024
        )
        
        if response.stop_reason == "end_turn":
            break
            
        if response.stop_reason == "tool_use":
            tool_result = await execute_tool(response.tool_use)
            state.history.append(tool_result_as_message)
            log_tool_call(state.investigation_id, response.tool_use, tool_result)
            
            # Agent evaluates whether to continue or conclude
            should_continue = await llm.should_continue(state.history)
            if not should_continue:
                break
    
    return generate_report(state)
```

### 3.4 Threat Intelligence Integration Pattern

Each integration follows the same adapter pattern:

```python
class ThreatIntelClient(ABC):
    @abstractmethod
    async def search(self, indicator: str, indicator_type: str) -> ToolResult
    @property
    @abstractmethod
    def service_name(self) -> str
    
    # Common: rate limiting, retry with exponential backoff, credential from env
```

Fallback strategy: if an API is unreachable, record `"unavailable"` in evidence with source标注, never infer "safe".

### 3.5 Error Handling & Retry Logic

| Failure Mode | Strategy |
|---|---|
| Tool execution timeout (>30s) | Log error, mark result inconclusive, continue |
| External API 429 (rate limit) | Exponential backoff, max 3 retries, then skip |
| External API 5xx | Same as above |
| LLM rate limit / token cap | Retry once with smaller context; escalate if fails |
| Invalid tool arguments | Reject at validation layer, log to execution monitor |
| Missing credentials | Tool returns `"not_configured"`, agent skips that tool |

### 3.6 Risk Scoring Algorithm

Evidence-based scoring (never purely LLM-driven):

```
base_score = 0
for each confirmed_evidence_item:
    base_score += evidence_weight * confirmation_strength
for each suggested_evidence_item:
    base_score += evidence_weight * 0.5 * confirmation_strength

if base_score >= CRITICAL_THRESHOLD:   risk = "Critical"
elif base_score >= HIGH_THRESHOLD:     risk = "High"
elif base_score >= MEDIUM_THRESHOLD:   risk = "Medium"
elif base_score > 0:                   risk = "Low"
else:                                 risk = "Inconclusive"
```

Thresholds are configurable constants, not learned parameters.

---

## 4. Scale and Reliability

### Load Estimation
- Target: 1–10 concurrent investigations per analyst session
- Each investigation: 3–15 tool calls, avg 2s each = 6–30s wall time
- SQLite handles this trivially; no queue needed

### Horizontal vs Vertical
- Desktop/single-server deployment. No clustering required for MVP.
- SQLite WAL mode enables safe concurrent reads.

### Monitoring
- Every tool call logged with timing to SQLite
- Streamlit "Execution Logs" tab shows real-time trace
- Evaluation dashboard aggregates across runs

### Failure Modes to Watch
- API key expiry → tool returns `not_configured`, investigate continues with remaining tools
- Network partition → threat intel tools fail gracefully, classification still works locally
- Dataset drift → versioned manifests ensure reproducibility

---

## 5. Trade-off Analysis

| Decision | Option A | Option B | Chosen | Rationale |
|---|---|---|---|---|
| LLM API | OpenAI | Cohere V2 (Tiny Aya) | **Cohere V2** | Project spec requires Tiny Aya Global; tool-calling well-supported |
| Persistence | PostgreSQL | SQLite | **SQLite** | Single-analyst app; zero ops overhead; file is portable |
| Dataset storage | Download at runtime | Bundle in repo | **Download script + local cache** | Repo stays small; reproducible via script |
| Agentic loop | ReAct | Plan-and-execute | **Hybrid** (planner extracts indicators, then tool-call loop) | Cleaner separation of concerns |
| Frontend | Streamlit | Gradio | **Streamlit** | Spec requirement; richer charting via Plotly |
| Threat intel | HTTP requests | MCP servers | **Direct HTTP** | Simpler deploys; avoids extra dependency; per spec |
| PDF generation | WeasyPrint | ReportLab | **WeasyPrint** | HTML-to-PDF is faster to iterate during hackathon |

### What to Revisit as System Grows
1. **SQLite → PostgreSQL**: If multi-user or cloud deployment becomes necessary.
2. **Direct HTTP → dedicated SDK wrappers**: As more threat-intel providers are added, shared abstractions reduce duplication.
3. **Hardcoded thresholds → tuned thresholds**: Current risk thresholds are heuristic; evaluation data will reveal better values.
4. **Single-threaded agent loop → async/concurrent tool calls**: If investigation latency becomes a bottleneck, parallelize independent tool calls.
5. **No caching layer**: Currently every API call hits the network. A Redis/memcached layer would help at scale.

---

## 6. Implementation Status

| Module | Status | Evidence |
|---|---|---|
| Email Threat Analyzer | ✅ **Implemented & tested** | `tools/email_analyzer.py` + `models/email_classifier.joblib` (97.5% test accuracy) |
| URL Investigation | ✅ **Implemented & tested** | `tools/url_analyzer.py` + `models/url_classifier.joblib` (99.6% test accuracy) |
| IP & Domain Intelligence | ✅ **Implemented** | `tools/ip_analyzer.py` + WHOIS/DNS integration |
| Network Log Analyzer | ✅ **Implemented** | `tools/network_analyzer.py` (CIC-IDS2017 CSV parsing) |
| Threat Intel (4 sources) | ✅ **Implemented** | URLhaus, PhishTank, AbuseIPDB, VirusTotal clients with caching |
| Tiny Aya LLM Client | ✅ **Implemented** | Cohere V2 SDK (`cohere.ClientV2`) with tool calling |
| Agentic Investigation | ✅ **Implemented** | 12-step tool-calling loop with evidence gate |
| Threat Correlation | ✅ **Implemented** | LLM-driven mid-loop + final correlation |
| Risk Assessment | ✅ **Implemented** | Evidence-weighted scoring with configurable thresholds |
| Execution Monitoring | ✅ **Implemented** | `audit_log` + `tool_calls` tables, Streamlit Logs tab |
| Evaluation Dashboard | ✅ **Implemented** | 7 metrics, 8 scenarios in `evaluation/benchmark.py` |
| Human Review & Reports | ✅ **Implemented** | Approve/Reject/Escalate, PDF generation |
| Tiny Aya LLM Client | ✅ **Implemented** | `agents/llm_client.py` — Cohere V2 SDK (`cohere.ClientV2`) |
| 4 Model Variants | ✅ **Configured** | `tiny-aya-global`, `tiny-aya-fire`, `tiny-aya-earth`, `tiny-aya-water` |
| Dataset Pipeline | ✅ **Implemented** | Full Section 8 pipeline (download→stats→version) |
| Evaluation Dataset | ✅ **Created** | 8 scenarios in `datasets/evaluation/scenarios.json` |
| Tests | ✅ **Passing** | 7/7 unit tests pass |

### Not Yet Verified (Requires API Keys)

| Feature | Status | Blocker |
|---|---|---|
| Tiny Aya native tool calling | ⚠️ **Not API-tested** | Requires valid Tiny Aya API key |
| Network classifier training | ⚠️ **Code ready, not trained** | No CIC-IDS2017 CSVs available locally |
| AgentTune fine-tuning | 📝 **Planned** | Framework ready, needs baseline trajectories |
| AuditKIT evaluation | 📝 **Framework ready** | Code ready, needs baseline + improved runs |
| Network classifier test accuracy | ❓ **Unknown** | No CIC-IDS2017 data to train/test |

---

## 7. Testing & Verification

### Unit Tests (7/7 passing)
```bash
pytest tests/ -v
```

### Integration Tests Needed
| Test | Status |
|---|---|
| Cohere V2 response parsing | ✅ Code ready |
| Native tool-call handling | ✅ Code ready (needs API key) |
| Argument validation | ✅ Code ready |
| Unknown tool rejection | ✅ Code ready |
| Tool-result message formatting | ✅ Code ready |
| API failure handling | ✅ Code ready (needs API key) |
| Final conclusion parsing | ✅ Code ready |
| Audit-log completeness | ✅ Code ready |

### Real API Smoke Test
```bash
# Requires valid TINY_AYA_API_KEY in .env
cd C:\Users\karth\Downloads\ThreatLens_AI
streamlit run app.py
```

---

## Appendix: Module-to-File Mapping

| Module | File(s) |
|---|---|
| Email Threat Analyzer | `tools/email_analyzer.py` |
| URL Investigation | `tools/url_analyzer.py` |
| IP & Domain Intelligence | `tools/ip_analyzer.py` |
| Network Log Analyzer | `tools/network_analyzer.py` |
| Tool Registry | `tools/tool_registry.py` |
| Investigation Agent | `agents/investigation_agent.py` |
| Indicator Planner | `agents/planner.py` |
| Response Generator | `agents/response_generator.py` |
| LLM Client | `agents/llm_client.py` |
| Threat Intel: URLhaus | `integrations/threat_intelligence/urlhaus.py` |
| Threat Intel: PhishTank | `integrations/threat_intelligence/phishtank.py` |
| Threat Intel: AbuseIPDB | `integrations/threat_intelligence/abuseipdb.py` |
| Threat Intel: VirusTotal | `integrations/threat_intelligence/virustotal.py` |
| Database | `database/db.py` + `database/schema.sql` |
| Dataset Pipeline | `data_pipeline/preprocess.py` |
| Evaluation | `evaluation/metrics.py` + `evaluation/benchmark.py` |
| Incident Reports | `reports/report_generator.py` |
| Dashboard | `app.py` |
| Config | `config/risk_thresholds.py` + `config/models.py` |
| Docs | `SYSTEM_DESIGN.md` + `docs/project_documentation.md` |