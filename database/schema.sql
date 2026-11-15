-- ThreatLens AI Database Schema

CREATE TABLE IF NOT EXISTS investigations (
    id TEXT PRIMARY KEY,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    type TEXT NOT NULL CHECK(type IN ('email', 'url', 'ip', 'network', 'multi')),
    input_content TEXT NOT NULL,
    risk_level TEXT,
    confidence REAL,
    summary TEXT,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'running', 'completed', 'escalated')),
    human_approved BOOLEAN,
    reviewer_note TEXT
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    step_order INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    arguments TEXT,
    result TEXT,
    duration_ms INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evidence_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    indicator_type TEXT NOT NULL CHECK(indicator_type IN ('email', 'url', 'ip', 'domain')),
    indicator_value TEXT NOT NULL,
    source_tool TEXT,
    source_record TEXT,
    veracity TEXT DEFAULT 'unknown' CHECK(veracity IN ('confirmed', 'suggested', 'unknown')),
    confidence REAL
);

CREATE TABLE IF NOT EXISTS correlations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    indicator_a TEXT NOT NULL,
    indicator_b TEXT NOT NULL,
    relationship TEXT NOT NULL,
    strength REAL
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_version TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    tool_selection_correct BOOLEAN,
    argument_accuracy REAL,
    task_completed BOOLEAN,
    classification_correct BOOLEAN,
    evidence_grounding REAL,
    tool_call_count INTEGER,
    escalation_appropriate BOOLEAN,
    actual_output TEXT,
    reference_output TEXT,
    completed_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Audit log for agent decisions
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    step_order INTEGER NOT NULL,
    event_type TEXT NOT NULL,  -- 'llm_decision', 'tool_call', 'tool_result', 'correlation', 'risk_assessment', 'escalation'
    payload TEXT,  -- JSON
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Review records for human oversight
CREATE TABLE IF NOT EXISTS review_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    decision TEXT NOT NULL CHECK(decision IN ('approved', 'rejected', 'escalated')),
    note TEXT,
    reviewer TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Model version tracking
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    metrics_json TEXT,  -- JSON string of evaluation metrics
    trained_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_investigations_created ON investigations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_investigations_status ON investigations(status);
CREATE INDEX IF NOT EXISTS idx_tool_calls_investigation ON tool_calls(investigation_id);
CREATE INDEX IF NOT EXISTS idx_evidence_investigation ON evidence_items(investigation_id);
CREATE INDEX IF NOT EXISTS idx_correlations_investigation ON correlations(investigation_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_scenario ON evaluation_runs(scenario_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_investigation ON audit_log(investigation_id);
CREATE INDEX IF NOT EXISTS idx_review_records_investigation ON review_records(investigation_id);