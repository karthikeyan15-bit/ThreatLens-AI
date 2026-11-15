"""URL feature transformers for ML pipeline - module-level for pickling."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
import numpy as np
import re
import tldextract
from urllib.parse import urlparse


SUSPICIOUS_TLDS = {".xyz", ".top", ".club", ".work", ".site", ".buzz", ".racing", ".gq", ".ml", ".cf", ".tk"}
URL_SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly"}


def extract_url_features(url: str) -> dict:
    """Extract structural features from a URL."""
    features = {}

    try:
        parsed = urlparse(url)
        tld_ext = tldextract.extract(parsed.netloc)
    except Exception:
        return _zero_features()

    hostname = parsed.netloc
    path = parsed.path
    query = parsed.query

    # Basic structural features
    features["url_length"] = len(url)
    features["hostname_length"] = len(hostname)
    features["path_length"] = len(path)
    features["query_length"] = len(query)

    # Scheme
    features["is_https"] = 1 if parsed.scheme == "https" else 0

    # IP-based hostname
    ip_match = re.match(r"^\d+\.\d+\.\d+\.\d+$", hostname.split(":")[0])
    features["is_ip_hostname"] = 1 if ip_match else 0

    # TLD features
    suffix = tld_ext.suffix
    features["tld_suspicious"] = 1 if suffix in {s.lstrip(".") for s in SUSPICIOUS_TLDS} else 0
    features["tld_length"] = len(suffix)

    # Subdomain features
    subdomain = tld_ext.subdomain
    features["subdomain_count"] = len(subdomain.split(".")) if subdomain else 0
    features["subdomain_length"] = len(subdomain)

    # Domain features
    registered_domain = tld_ext.top_domain_under_public_suffix
    features["domain_length"] = len(registered_domain)

    # URL shortener
    features["is_shortener"] = 1 if any(s in hostname for s in URL_SHORTENERS) else 0

    # Path patterns
    features["path_has_exe"] = 1 if re.search(r"\.(exe|bat|cmd|scr|js|vbs|ps1|dll|com)(\?|$)", path) else 0
    features["path_has_redirect"] = 1 if re.search(r"[?&](redirect|return|next|r)=https?://", query) else 0
    features["path_has_login"] = 1 if re.search(r"/login", path) else 0
    features["path_has_wp_login"] = 1 if "/wp-login.php" in path else 0
    features["path_depth"] = len([p for p in path.split("/") if p])

    # Query features
    features["query_param_count"] = len(query.split("&")) if query else 0

    # Special characters
    features["at_symbol_count"] = url.count("@")
    features["dash_count"] = hostname.count("-")
    features["underscore_count"] = url.count("_")
    features["percent_encoding"] = 1 if "%" in url else 0

    # IDN/homograph
    features["has_idn"] = 1 if "xn--" in hostname else 0

    # Entropy of hostname
    if hostname:
        char_counts = {}
        for c in hostname:
            char_counts[c] = char_counts.get(c, 0) + 1
        entropy = -sum((count/len(hostname)) * np.log2(count/len(hostname)) for count in char_counts.values())
        features["hostname_entropy"] = entropy
    else:
        features["hostname_entropy"] = 0

    return features


def _zero_features() -> dict:
    """Return zero features for failed parses."""
    keys = [
        "url_length", "hostname_length", "path_length", "query_length",
        "is_https", "is_ip_hostname", "tld_suspicious", "tld_length",
        "subdomain_count", "subdomain_length", "domain_length",
        "is_shortener", "path_has_exe", "path_has_redirect", "path_has_login",
        "path_has_wp_login", "path_depth", "query_param_count",
        "at_symbol_count", "dash_count", "underscore_count",
        "percent_encoding", "has_idn", "hostname_entropy"
    ]
    return {k: 0 for k in keys}


class StructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract structural features from URLs."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for url in X:
            feat = extract_url_features(url)
            features.append([feat.get(k, 0) for k in self.feature_names_])
        return np.array(features)

    def fit(self, X, y=None):
        # Determine feature names from first sample
        if len(X) > 0:
            sample_features = extract_url_features(X[0])
            self.feature_names_ = list(sample_features.keys())
        return self


class URLTFIDFExtractor(BaseEstimator, TransformerMixin):
    """Extract char-level TF-IDF features from URLs."""
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            analyzer="char",
            ngram_range=(2, 4),
            sublinear_tf=True
        )

    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X)