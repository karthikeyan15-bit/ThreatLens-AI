"""ThreatLens AI — Agentic Cybersecurity Investigation Platform."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import plotly.express as px

from dotenv import load_dotenv

from database.db import (
    get_investigations,
    get_tool_calls,
    get_evidence,
    get_correlations,
    get_investigation,
    init_db,
    update_investigation,
)
from agents.investigation_agent import run_investigation
from reports.report_generator import generate_html_report, save_pdf
from evaluation.benchmark import run_all_scenarios


load_dotenv()

DB_PATH = os.getenv("DATABASE_PATH", "database/threatlens.db")

if not Path(DB_PATH).parent.exists():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
init_db(DB_PATH)

TAB_NAMES = [
    "Overview", "Investigate", "Email", "URL Scanner", "IP Intelligence",
    "Network Analyzer", "Threat Intelligence", "Execution Logs", "Evaluation", "Review & Reports",
]


def render_overview():
    investigations = get_investigations(limit=20)
    total = len(investigations)
    by_risk = {}
    for inv in investigations:
        level = inv["risk_level"] or "Unknown"
        by_risk[level] = by_risk.get(level, 0) + 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Investigations", total)
    c2.metric("Completed", sum(1 for i in investigations if i["status"] == "completed"))
    c3.metric("Escalated", sum(1 for i in investigations if i["status"] == "escalated"))
    c4.metric("With Risk Level", sum(1 for i in investigations if i["risk_level"]))

    if by_risk:
        fig = px.pie(
            names=list(by_risk.keys()), values=list(by_risk.values()),
            title="Risk Distribution",
            color_discrete_map={
                "Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336",
                "Critical": "#B71C1C", "Inconclusive": "#9E9E9E",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Investigations")
    if investigations:
        rows = [[i["id"], i["type"], i["risk_level"], i["status"], i["created_at"][:19]]
                for i in investigations]
        st.table(rows)
    else:
        st.info("No investigations yet. Go to **Investigate** to start one.")


def render_investigate():
    selected_id = st.session_state.get("selected_inv_id")

    if selected_id:
        inv = get_investigation(selected_id)
        if inv:
            st.header(f"Investigation: {inv['id']}")
            st.caption(f"**Type:** {inv['type']}  |  **Status:** {inv['status']}  |  **Created:** {inv['created_at'][:19]}")

            risk_colors = {"Low": "green", "Medium": "orange", "High": "red", "Critical": "darkred", "Inconclusive": "grey"}
            rc = risk_colors.get(inv["risk_level"], "grey")
            col_a, col_b = st.columns([2, 1])
            with col_a:
                content_preview = inv["input_content"][:200] + "…" if len(inv["input_content"]) > 200 else inv["input_content"]
                st.markdown(f"**Input:** `{content_preview}`")
            with col_b:
                st.markdown(f"**Risk:** :{rc}**[{inv['risk_level'] or 'N/A'}]**")

            tool_calls = get_tool_calls(selected_id)
            evidence = get_evidence(selected_id)
            correlations = get_correlations(selected_id)

            if tool_calls:
                st.subheader("Tool Execution Trace")
                for tc in tool_calls:
                    with st.expander(f"Step {tc['step_order']}: `{tc['tool_name']}` ({tc['duration_ms']}ms)", expanded=False):
                        st.json(tc.get("arguments", {}))
                        result = tc.get("result")
                        if isinstance(result, str):
                            try:
                                result = json.loads(result)
                            except Exception:
                                pass
                        st.json(result)

            if evidence:
                st.subheader(f"Evidence ({len(evidence)} items)")
                for e in evidence:
                    badge = "✅ Confirmed" if e["veracity"] == "confirmed" else "⚠️ Suggested"
                    st.markdown(f"- **{badge}** [{e['indicator_type']}] `{e['indicator_value']}` — *{e['source_tool']}*")

            if correlations:
                st.subheader(f"Correlations ({len(correlations)})")
                for c in correlations:
                    st.write(f"- `{c['indicator_a']}` ↔ `{c['indicator_b']}` — {c['relationship']} ({round(c['strength']*100)}%)")

            st.divider()
            if st.button("← Back to new investigation", key="back_investigate"):
                st.session_state.selected_inv_id = None
                st.rerun()
            return

    st.header("🔬 AI-Powered Investigation")
    st.caption("Submit an incident description and the agent will investigate using multiple tools.")

    request = st.text_area(
        "Investigation Request",
        placeholder=("Describe the security incident, paste suspicious email content, "
                     "provide URLs/IPs to investigate, or upload network log data…"),
        height=150,
    )

    if st.button("🔍 Run Investigation", type="primary", disabled=not request.strip()):
        import os as _os
        if not _os.getenv("HF_TOKEN"):
            st.error("HF_TOKEN is not configured. Please set it in Hugging Face Space Secrets.")
        else:
            with st.spinner("Agent is investigating — this may take up to 60 seconds…"):
                t0 = time.time()
                result = run_investigation(request.strip())
                elapsed = time.time() - t0

            if result.get("error"):
                st.error(f"Investigation failed: {result.get('error')}")
            else:
                st.success(f"✅ Investigation complete in {elapsed:.1f}s")
                st.session_state.last_result = result
                st.session_state.selected_inv_id = result.get("investigation_id")
                st.rerun()

    if st.session_state.get("last_result"):
        result = st.session_state.last_result
        inv_id = result.get("investigation_id", "")
        st.subheader(f"Result — {inv_id}")

        cols = st.columns(3)
        cols[0].metric("Risk Level", result.get("risk_level", "N/A"))
        cols[1].metric("Confidence", f"{result.get('confidence', 0):.0%}")
        cols[2].metric("Steps Taken", result.get("steps_taken", 0))

        st.markdown(f"**Summary:**\n\n{result.get('summary', 'No summary generated.')}")

        if result.get("escalation_required"):
            st.warning("⚠️ This investigation requires human escalation.")

        with st.expander("Full Tool Trace", expanded=False):
            for tc in result.get("tool_calls", []):
                st.json(tc)

        st.session_state.selected_inv_id = inv_id


def render_email_analyzer():
    st.header("📧 Email Threat Analyzer")
    mode = st.radio("Input mode:", ["Paste email content", "Upload .eml / .txt file"], horizontal=True)

    email_content = ""
    if mode == "Paste email content":
        email_content = st.text_area("Email content", height=200, placeholder="Paste the full email text here…")
    else:
        uploaded = st.file_uploader("Upload email file", type=["eml", "txt", "msg"])
        if uploaded:
            email_content = uploaded.read().decode("utf-8", errors="ignore")

    if st.button("Analyze Email", disabled=not email_content.strip()):
        from tools.email_analyzer import classify_email
        result = classify_email(email_content)

        st.subheader("Classification Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Classification", result["classification"].capitalize())
        c2.metric("Phishing Score", f"{result['score']:.2f}")
        c3.metric("Confidence", f"{result['confidence']:.0%}")

        if result["headers"]:
            with st.expander("Extracted Headers"):
                st.json(result["headers"])

        if result["extracted_urls"]:
            st.markdown("**Extracted URLs:**")
            for u in result["extracted_urls"]:
                st.code(u)

        if result["extracted_domains"]:
            st.markdown("**Extracted Domains:**")
            for d in result["extracted_domains"]:
                st.write(f"- {d}")


def render_url_scanner():
    st.header("🔗 URL Investigator")
    url = st.text_input("Enter URL to investigate", placeholder="https://suspicious-site.example.com/path")

    if st.button("Analyze URL", disabled=not url.strip()):
        from tools.url_analyzer import analyze_url
        result = analyze_url(url.strip())

        st.subheader("Structural Analysis")
        c1, c2 = st.columns(2)
        c1.metric("Risk Score", f"{result['structural_risk_score']:.2f}")
        c2.metric("Verdict", result["combined_verdict"].capitalize())

        if result["structural_indicators"]:
            st.markdown("**Indicators Found:**")
            for ind in result["structural_indicators"]:
                st.warning(f"⚠ {ind}")

        with st.expander("Parsed URL Details"):
            st.json(result["parsed"])


def render_ip_intelligence():
    st.header("🔍 IP Intelligence")
    indicator = st.text_input("IP Address or Domain")
    ind_type = st.selectbox("Type", ["ip", "domain"], key="ip_intel_type_select")

    if st.button("Analyze", disabled=not indicator.strip()):
        from tools.ip_analyzer import run as ip_run
        result = ip_run({"indicator": indicator.strip()})
        if "error" in result:
            st.error(result["error"])
        else:
            st.json(result)


def render_network_analyzer():
    st.header("🌐 Network Log Analyzer")
    uploaded = st.file_uploader("Upload CSV (CIC-IDS2017 format)", type=["csv"])
    if uploaded:
        csv_content = uploaded.read().decode("utf-8", errors="ignore")
        if st.button("Analyze Network Log"):
            from tools.network_analyzer import analyze_network_log
            result = analyze_network_log(csv_content)
            st.metric("Overall Risk", result["overall_risk"].capitalize())
            st.json(result)


def render_threat_intelligence():
    st.header("🛡️ Threat Intelligence Lookup")
    st.caption("Query live threat intelligence sources (URLhaus, OpenPhish, AbuseIPDB, VirusTotal)")

    indicator = st.text_input("Indicator (URL, IP, or Domain)")
    ind_type = st.selectbox("Type", ["url", "ip", "domain"], key="threat_intel_type_select")

    if st.button("Lookup", disabled=not indicator.strip()):
        from tools.threat_intel_tool import run as ti_run
        result = ti_run({"indicator": indicator.strip(), "indicator_type": ind_type})

        st.subheader("Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Verdict", result.get("overall_verdict", "unknown"))
        col2.metric("Sources Queried", result.get("sources_queried", 0))
        col3.metric("Sources Unavailable", result.get("sources_unavailable", 0))

        if result.get("safety_note"):
            st.warning(result["safety_note"])

        for r in result.get("results", []):
            with st.expander(f"{r.get('service', 'unknown')} — {r.get('verdict', 'unknown')}"):
                st.json(r)


def render_execution_logs():
    st.header("📋 Execution Logs")
    inv_id = st.text_input("Investigation ID")
    if not inv_id:
        recent = get_investigations(limit=10)
        inv_id = st.selectbox("Or select recent", [i["id"] for i in recent], index=None, key="exec_logs_recent_select")

    if inv_id:
        tool_calls = get_tool_calls(inv_id)
        evidence = get_evidence(inv_id)
        cols = st.columns(2)
        with cols[0]:
            st.subheader(f"Tool Calls ({len(tool_calls)})")
            for tc in tool_calls:
                st.markdown(f"**Step {tc['step_order']}:** `{tc['tool_name']}` ({tc['duration_ms']}ms)")
                st.json(tc.get("arguments", {}))
        with cols[1]:
            st.subheader(f"Evidence ({len(evidence)})")
            for e in evidence:
                st.markdown(f"- [{e['veracity']}] {e['indicator_type']}: `{e['indicator_value']}`")


def render_evaluation():
    st.header("📊 Agent Evaluation Dashboard")

    c1, c2 = st.columns(2)
    with c1:
        version = st.selectbox("Agent Version", ["baseline", "improved"], key="eval_version_select")
    with c2:
        if st.button("Run Benchmark"):
            with st.spinner("Running evaluation scenarios…"):
                result = run_all_scenarios(agent_version=version)
            st.session_state.eval_result = result

    if st.session_state.get("eval_result"):
        res = st.session_state.eval_result
        if "error" in res:
            st.error(res["error"])
        else:
            m = res["aggregated_metrics"]
            st.subheader(f"Results — {res['agent_version']} agent ({m['total_runs']} scenarios)")
            mc = st.columns(4)
            mc[0].metric("Tool Selection Accuracy", f"{m.get('tool_selection_accuracy', 0):.1%}")
            mc[1].metric("Avg Argument Accuracy", f"{m.get('avg_argument_accuracy', 0):.1%}")
            mc[2].metric("Completion Rate", f"{m.get('completion_rate', 0):.1%}")
            mc[3].metric("Avg Tool Calls", f"{m.get('avg_tool_calls', 0):.1f}")
            st.metric("Classification Accuracy", f"{m.get('classification_accuracy', 0):.1%}")
            st.metric("Evidence Grounding", f"{m.get('avg_evidence_grounding', 0):.1%}")
            st.metric("Escalation Accuracy", f"{m.get('escalation_accuracy', 0):.1%}")

            per_scenario = res.get("per_scenario", [])
            if per_scenario:
                rows = [[r["scenario_id"], r["metrics"]["task_completed"],
                         r["metrics"]["classification_correct"],
                         r["metrics"]["tool_call_count"]] for r in per_scenario]
                st.table(rows)


def render_review_reports():
    st.header("👁️ Human Review & Incident Reports")

    inv_id = st.text_input("Investigation ID for review")
    if not inv_id:
        recent = get_investigations(limit=10)
        inv_id = st.selectbox("Or select recent", [i["id"] for i in recent], index=None, key="review_reports_recent_select")

    if inv_id:
        inv = get_investigation(inv_id)
        if not inv:
            st.error("Investigation not found.")
            return

        st.subheader(f"Investigation: {inv['id']}")
        risk_colors = {"Low": "green", "Medium": "orange", "High": "red", "Critical": "darkred"}
        rc = risk_colors.get(inv["risk_level"], "grey")
        st.markdown(f"**Risk:** :{rc}**[{inv['risk_level'] or 'N/A'}]**  |  **Status:** {inv['status']}")
        st.markdown(f"**Summary:** {inv.get('summary', 'N/A')}")

        evidence = get_evidence(inv_id)
        correlations = get_correlations(inv_id)

        if evidence:
            st.subheader("Evidence")
            for e in evidence:
                badge = "✅ Confirmed" if e["veracity"] == "confirmed" else "⚠️ Suggested"
                st.markdown(f"- **{badge}** [{e['indicator_type']}] `{e['indicator_value']}`")

        if correlations:
            st.subheader("Correlations")
            for c in correlations:
                st.write(f"- `{c['indicator_a']}` ↔ `{c['indicator_b']}` ({c['relationship']}, {round(c['strength']*100)}%)")

        analyst_note = st.text_area("Analyst Note", placeholder="Add your review comments…")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Approve", type="primary"):
                update_investigation(inv_id, human_approved=True, reviewer_note=analyst_note)
                st.success("Approved.")
        with c2:
            if st.button("Reject"):
                update_investigation(inv_id, human_approved=False, reviewer_note=analyst_note)
                st.warning("Rejected.")
        with c3:
            if st.button("📄 Generate Report"):
                html = generate_html_report(
                    {"risk_level": inv.get("risk_level", "Unknown"),
                     "confidence": inv.get("confidence", 0),
                     "summary": inv.get("summary", ""),
                     "evidence_items": evidence,
                     "correlations": correlations,
                     "tool_calls_made": len(get_tool_calls(inv_id)),
                     "tool_calls": get_tool_calls(inv_id),
                     "evidence_count": len(evidence),
                     "correlation_count": len(correlations),
                     },
                    inv_id,
                )
                ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                pdf_path = f"reports/{inv_id}_{ts}.pdf"
                html_path = f"reports/{inv_id}_{ts}.html"
                result = save_pdf(html, pdf_path)
                # Fallback: if PDF failed, offer the HTML report
                download_path = result.get("path") or html_path
                ext = "pdf" if download_path.endswith(".pdf") else "html"
                mime_type = "application/pdf" if ext == "pdf" else "text/html"
                with open(download_path, "rb") as f:
                    st.download_button(
                        "Download Report", data=f,
                        file_name=f"{inv_id}_report.{ext}", mime=mime_type,
                    )
                if not pdf_path.endswith(".pdf") and result.get("note"):
                    st.info(result["note"])


# ── Main ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ ThreatLens AI")
    st.caption("Agentic Cybersecurity Investigation Platform")
    st.divider()

    if st.button("New Investigation"):
        st.session_state.selected_inv_id = None
        st.session_state.last_result = None
        st.session_state.current_tab = "investigate"
        st.rerun()

    st.divider()
    st.markdown("**Recent Investigations**")
    recent = get_investigations(limit=5)
    for inv in recent:
        risk_icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "Critical": "🟣", "Inconclusive": "⚪"}.get(inv["risk_level"], "⚪")
        if st.button(f"{risk_icon} {inv['id']}", key=f"nav_{inv['id']}"):
            st.session_state.selected_inv_id = inv["id"]
            st.rerun()


tabs = st.tabs(TAB_NAMES)

with tabs[0]:
    render_overview()

with tabs[1]:
    render_investigate()

with tabs[2]:
    render_email_analyzer()

with tabs[3]:
    render_url_scanner()

with tabs[4]:
    render_ip_intelligence()

with tabs[5]:
    render_network_analyzer()

with tabs[6]:
    render_threat_intelligence()

with tabs[7]:
    render_execution_logs()

with tabs[8]:
    render_evaluation()

with tabs[9]:
    render_review_reports()
