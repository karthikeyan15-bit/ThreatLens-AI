"""Lightweight test verifying the 10 Streamlit tabs map to their renderers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import only what we need — avoid triggering full app startup
import app


def test_tab_names_count():
    assert len(app.TAB_NAMES) == 10


def test_no_broken_selection_state():
    """The old idx_map / current / tab_idx conditional logic must be gone."""
    # These symbols should no longer exist in the module
    assert not hasattr(app, "idx_map"), "idx_map should have been removed"
    assert not hasattr(app, "tab_idx"), "tab_idx should have been removed"


def test_all_ten_renderers_are_callable_functions():
    expected = [
        ("Overview", app.render_overview),
        ("Investigate", app.render_investigate),
        ("Email", app.render_email_analyzer),
        ("URL Scanner", app.render_url_scanner),
        ("IP Intelligence", app.render_ip_intelligence),
        ("Network Analyzer", app.render_network_analyzer),
        ("Threat Intelligence", app.render_threat_intelligence),
        ("Execution Logs", app.render_execution_logs),
        ("Evaluation", app.render_evaluation),
        ("Review & Reports", app.render_review_reports),
    ]
    names = [n for n, _ in expected]
    funcs = [f for _, f in expected]
    for name, fn in expected:
        assert callable(fn), f"{name} does not map to a callable"
    assert all(n in app.TAB_NAMES for n in names), "Some expected tab names are missing"
    assert set(app.TAB_NAMES) == set(names), f"Tab names mismatch: {set(app.TAB_NAMES)} != {set(names)}"
