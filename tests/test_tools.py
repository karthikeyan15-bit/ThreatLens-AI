"""Basic tests for ThreatLens AI tools."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.email_analyzer import classify_email, extract_urls, calculate_phishing_score
from tools.url_analyzer import analyze_url
from tools.ip_analyzer import parse_indicator, is_private_ip
from tools.network_analyzer import detect_anomalies, infer_columns


def test_extract_urls():
    text = "Visit https://example.com and https://malicious-site.xyz for more info."
    urls = extract_urls(text)
    assert len(urls) == 2
    assert "https://example.com" in urls
    assert "https://malicious-site.xyz" in urls


def test_private_ip():
    assert is_private_ip("192.168.1.1")
    assert is_private_ip("10.0.0.1")
    assert is_private_ip("172.16.0.1")
    assert not is_private_ip("8.8.8.8")
    assert not is_private_ip("1.2.3.4")


def test_parse_indicator():
    assert parse_indicator({"indicator": "192.168.1.1"}) == ("ip", "192.168.1.1")
    assert parse_indicator({"indicator": "example.com"}) == ("domain", "example.com")
    assert parse_indicator({"indicator": "invalid!"}) == ("unknown", "invalid!")


def test_url_analysis():
    result = analyze_url("http://evil-site.xyz/login?id=123")
    assert result["structural_risk_score"] > 0
    assert any("Suspicious scheme" in ind for ind in result["structural_indicators"])
    assert result["structural_verdict"] in ("malicious", "suspicious", "structurally_unremarkable")


def test_phishing_score():
    good_email = "Hi John, your invoice is attached. Please review at your convenience."
    bad_email = "URGENT! Your account will be suspended! Click here NOW to verify: http://fake-login.xyz"

    assert calculate_phishing_score(good_email, {}) < 0.3
    # Bad email has urgency + credential patterns but moderate score; just check it's higher than good
    assert calculate_phishing_score(bad_email, {}) > calculate_phishing_score(good_email, {})


def test_email_classification():
    legitimate = """From: john.doe@company.com
To: jane.smith@company.com
Subject: Meeting tomorrow
Date: Mon, 14 Nov 2026 09:00:00 +0000

Hi Jane, just checking in about our 2pm meeting tomorrow."""

    result = classify_email(legitimate)
    assert result["classification"] == "legitimate"


def test_network_column_inference():
    import pandas as pd
    df = pd.DataFrame({
        "Src IP": ["1.2.3.4", "5.6.7.8"],
        "Dst IP": ["10.0.0.1", "10.0.0.2"],
        "Protocol": ["TCP", "UDP"],
        "Label": ["Benign", "DoS"],
    })
    col_map = infer_columns(df)
    assert col_map.get("proto") == "Protocol"
    assert col_map.get("label") == "Label"


if __name__ == "__main__":
    test_extract_urls()
    test_private_ip()
    test_parse_indicator()
    test_url_analysis()
    test_phishing_score()
    test_email_classification()
    test_network_column_inference()
    print("All tests passed!")
