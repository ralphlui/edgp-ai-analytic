"""
Responsible AI helpers: detect and redact sensitive content from model outputs.

Features:
- PII redaction: emails, SSNs, conservative phone numbers
- Secret redaction: OpenAI keys, AWS access keys, JWTs, URL basic-auth, PEM blocks, long base64 blobs
- Configurable toggles and placeholder, with per-category counts

Notes:
- Heuristics can produce false positives/negatives.
- Favor precision for AWS/JWT/URL; keep phone pattern conservative.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Pattern, Tuple


# -------------------------
# Compiled patterns (bounded)
# -------------------------

# PII
EMAIL_RE: Pattern[str] = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
SSN_RE: Pattern[str] = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Conservative US-like phone numbers; requires separators; avoids matching long digit blobs/dates
PHONE_RE: Pattern[str] = re.compile(
    r"""
    (?<!\d)                                  # no digit to the left
    (?:\+?\d{1,3}[-.\s])?                    # optional country code with sep
    (?:\(?\d{3}\)?[-.\s])                    # area code with sep
    \d{3}[-.\s]\d{4}                         # local number with sep
    (?!\d)                                   # no digit to the right
    """,
    re.VERBOSE,
)

# Secrets / tokens
OPENAI_KEY_RE: Pattern[str] = re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")  # typical OpenAI prefix

# AWS Access Key ID: typically AKIA/ASIA + 16 uppercase alphanumerics.
# Broaden slightly (12-20) to catch near-matches that tests and logs often include.
AWS_ACCESS_KEY_ID_RE: Pattern[str] = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{12,20}\b")

# JWT (three base64url-like segments with dots)
JWT_RE: Pattern[str] = re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")

# URL Basic Auth credentials: https://user:pass@host
URL_CRED_RE: Pattern[str] = re.compile(r"https?://[^/\s:@]+:[^@\s]+@[^/\s]+")

# PEM private key block (DOTALL via [\s\S])
PEM_PRIVATE_RE: Pattern[str] = re.compile(
    r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC )?PRIVATE KEY-----"
)

# Long base64-like strings; useful to hide large secrets/blobs
# NB: Keep threshold fairly high to avoid redacting normal text.
LONG_BASE64_RE: Pattern[str] = re.compile(r"\b[A-Za-z0-9+/]{200,}={0,2}\b")


@dataclass
class RedactionConfig:
    enable_pii_redaction: bool = True
    enable_secret_redaction: bool = True
    enable_jwt_redaction: bool = True
    enable_url_credential_redaction: bool = True
    enable_base64_redaction: bool = True
    base64_min_len: int = 200
    placeholder_fmt: str = "[REDACTED:{label}]"


def _redact(pattern: Pattern[str], text: str, label: str, stats: Dict[str, int], placeholder_fmt: str) -> str:
    """Single-pass redaction with per-label counts."""
    def repl(_: re.Match) -> str:
        stats[label] = stats.get(label, 0) + 1
        return placeholder_fmt.format(label=label)
    return pattern.sub(repl, text)


def apply_responsible_output_filters(
    text: str,
    cfg: RedactionConfig | None = None,
) -> Tuple[str, Dict[str, int]]:
    """
    Redact common PII and secrets in a model output.

    Returns:
        filtered_text: str
        redaction_stats: Dict[label, count]
    """
    if not text:
        return text, {}

    cfg = cfg or RedactionConfig()
    stats: Dict[str, int] = {}
    filtered = text

    # URL credentials first to avoid partial email redaction inside credentials
    if cfg.enable_url_credential_redaction:
        filtered = _redact(URL_CRED_RE, filtered, "URL_CREDENTIALS", stats, cfg.placeholder_fmt)

    # PII next
    if cfg.enable_pii_redaction:
        filtered = _redact(EMAIL_RE, filtered, "EMAIL", stats, cfg.placeholder_fmt)
        filtered = _redact(SSN_RE, filtered, "SSN", stats, cfg.placeholder_fmt)
        filtered = _redact(PHONE_RE, filtered, "PHONE", stats, cfg.placeholder_fmt)  # conservative to reduce false positives

    # Secrets
    if cfg.enable_secret_redaction:
        filtered = _redact(OPENAI_KEY_RE, filtered, "OPENAI_KEY", stats, cfg.placeholder_fmt)
        filtered = _redact(AWS_ACCESS_KEY_ID_RE, filtered, "AWS_ACCESS_KEY_ID", stats, cfg.placeholder_fmt)
        filtered = _redact(PEM_PRIVATE_RE, filtered, "PRIVATE_KEY", stats, cfg.placeholder_fmt)

    # JWT
    if cfg.enable_jwt_redaction:
        filtered = _redact(JWT_RE, filtered, "JWT", stats, cfg.placeholder_fmt)

    # (URL credentials already handled earlier)

    # Long base64 blobs
    if cfg.enable_base64_redaction and cfg.base64_min_len >= 50:
        # Compile a thresholded regex at runtime only if threshold differs from default.
        if cfg.base64_min_len != 200:
            dyn_re = re.compile(rf"\b[A-Za-z0-9+/]{{{cfg.base64_min_len},}}={{0,2}}\b")
            filtered = _redact(dyn_re, filtered, "BASE64", stats, cfg.placeholder_fmt)
        else:
            filtered = _redact(LONG_BASE64_RE, filtered, "BASE64", stats, cfg.placeholder_fmt)

    return filtered, stats
