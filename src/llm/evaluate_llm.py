"""
Automated evaluation of the LLM diagnostic layer (no human panel needed).

Three quantitative metrics, each a % over a sample of predictions:
  1) STRUCTURAL VALIDITY  — fraction of outputs that parse as JSON with all required fields
                            and a valid severity value. Measures integration reliability.
  2) GROUNDING RATE       — fraction whose likely_cause is grounded in the class's provided T1
                            cause (token overlap), i.e. NOT an invented cause. Anti-hallucination.
  3) FAITHFULNESS         — fraction whose severity is consistent with the measured scale+saliency
                            that were given as input (large+high-saliency should not be 'low', etc.).
                            Measures whether the model respects the facts it was given.

Report these three numbers directly. They are reproducible and defensible.
"""
from __future__ import annotations
import re


_STOP = {"a","an","the","of","to","and","or","is","are","in","on","by","with","due",
         "which","that","this","caused","shows","photo","surface","steel","defect","from"}


def _content_tokens(text: str):
    toks = re.findall(r"[a-z]+", (text or "").lower())
    return {t for t in toks if t not in _STOP and len(t) > 2}


def check_structural_validity(diag: dict) -> bool:
    return bool(diag.get("_valid"))


def check_grounding(diag: dict, t1_cause: str, min_overlap: int = 1) -> bool:
    """likely_cause should share content words with the provided T1 cause — i.e. it reasons
    from the given cause rather than inventing an unrelated one."""
    if not diag.get("likely_cause"):
        return False
    cause_toks = _content_tokens(diag["likely_cause"])
    t1_toks = _content_tokens(t1_cause)
    return len(cause_toks & t1_toks) >= min_overlap


def check_faithfulness(diag: dict, attributes: dict) -> bool:
    """severity must be consistent with the measured scale + saliency we handed the model."""
    sev = (diag.get("severity") or "").lower()
    if sev not in {"low", "moderate", "high"}:
        return False
    scale = attributes.get("Scale")
    sal = attributes.get("Saliency")
    # map measured facts to an allowed severity band; flag only clear contradictions
    big = scale in {"large", "extensive"}
    small = scale in {"tiny", "small", "none"}
    high_sal = sal == "high"
    low_sal = sal == "low"
    if big and high_sal and sev == "low":
        return False      # large + very salient can't be "low"
    if small and low_sal and sev == "high":
        return False      # tiny + faint can't be "high"
    return True


def evaluate_batch(records: list[dict]) -> dict:
    """records: list of {'diag':..., 't1_cause':..., 'attributes':...}. Returns the 3 metrics."""
    n = len(records)
    if n == 0:
        return {}
    valid = sum(check_structural_validity(r["diag"]) for r in records)
    grounded = sum(check_grounding(r["diag"], r["t1_cause"]) for r in records)
    faithful = sum(check_faithfulness(r["diag"], r["attributes"]) for r in records)
    return {
        "n": n,
        "structural_validity_pct": round(100 * valid / n, 1),
        "grounding_rate_pct": round(100 * grounded / n, 1),
        "faithfulness_pct": round(100 * faithful / n, 1),
    }
