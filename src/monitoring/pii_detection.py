"""
PII Detection — scans query, response, and contexts for sensitive data.
Supports three enforcement policies: block | mask | redact
"""
import re
import hashlib
from dataclasses import dataclass
from enum import Enum

from src.core.config import settings


class PIIPolicy(str, Enum):
    BLOCK  = "block"
    MASK   = "redact"
    REDACT = "redact"


@dataclass
class PIIMatch:
    pii_type:     str
    field:        str
    matched_text: str
    masked_value: str


# Patterns: (label, compiled_regex)
_PATTERNS = [
    ("email",       re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")),
    ("phone",       re.compile(r"\+?[\d\s\-().]{7,15}\d")),
    ("ssn",         re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("api_key",     re.compile(r"(?i)(api[_\-]?key|token|secret)[=:\s]+['\"]?[\w\-]{16,}")),
    ("password",    re.compile(r"(?i)(password|passwd|pwd)[=:\s]+['\"]?\S+")),
]

# Placeholder labels used when masking
_PLACEHOLDERS = {
    "email":       "[EMAIL]",
    "phone":       "[PHONE]",
    "ssn":         "[SSN]",
    "credit_card": "[CARD]",
    "api_key":     "[API_KEY]",
    "password":    "[PASSWORD]",
}


def _mask_value(value: str) -> str:
    if len(value) <= 4:
        return "***"
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


def _hash_value(value: str) -> str:
    return "[HASH:" + hashlib.sha256(value.encode()).hexdigest()[:8] + "]"


def scan_for_pii(query: str, response: str, contexts: list[str]) -> list[PIIMatch]:
    """Scan all fields for PII. Returns list of PIIMatch (empty = clean)."""
    findings: list[PIIMatch] = []
    fields = {
        "query":    query,
        "response": response,
        **{f"context[{i}]": ctx for i, ctx in enumerate(contexts or [])},
    }

    for field_name, text in fields.items():
        if not text:
            continue
        for pii_type, pattern in _PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group()
                findings.append(PIIMatch(
                    pii_type     = pii_type,
                    field        = field_name,
                    matched_text = raw,
                    masked_value = _mask_value(raw),
                ))

    return findings


def apply_pii_policy(text: str, policy: str | None = None) -> str:
    """
    Apply PII policy to a text string.

    policy:
        block  — caller should raise 422 before reaching this
        mask   — replace with [TYPE] placeholders
        redact — remove matched text entirely
    """
    effective_policy = policy or settings.PII_POLICY

    if effective_policy == "block":
        return text  # handled upstream

    result = text
    for pii_type, pattern in _PATTERNS:
        if effective_policy == "mask":
            result = pattern.sub(_PLACEHOLDERS.get(pii_type, "[PII]"), result)
        elif effective_policy == "redact":
            result = pattern.sub("", result)

    return result


def process_inputs_with_policy(
    query: str,
    response: str,
    contexts: list[str],
    policy: str | None = None,
) -> tuple[str, str, list[str], list[PIIMatch], str]:
    """
    Scan and apply PII policy.

    Returns:
        (clean_query, clean_response, clean_contexts, matches, action_taken)
    """
    effective = policy or settings.PII_POLICY
    matches   = scan_for_pii(query, response, contexts)

    if not matches:
        return query, response, contexts, [], "none"

    if effective == "block":
        return query, response, contexts, matches, "block"

    clean_query    = apply_pii_policy(query,    effective)
    clean_response = apply_pii_policy(response, effective)
    clean_contexts = [apply_pii_policy(c, effective) for c in contexts]

    return clean_query, clean_response, clean_contexts, matches, effective


def pii_summary(matches: list[PIIMatch]) -> dict:
    """Returns a JSON-serialisable summary of detected PII."""
    return {
        "pii_detected": len(matches) > 0,
        "count":        len(matches),
        "findings": [
            {"type": m.pii_type, "field": m.field, "masked": m.masked_value}
            for m in matches
        ],
    }