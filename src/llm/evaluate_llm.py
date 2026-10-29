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


# ---------------------------------------------------------------------------
# End-to-end evaluation: the LLM is fed the CLASSIFIER'S prediction, exactly as in the
# dashboard, so errors upstream of the LLM show up in the numbers.
# ---------------------------------------------------------------------------
def evaluate_pipeline(records: list[dict], t1: dict, review_threshold: float) -> dict:
    """records: one per sampled image, as written by notebook 04 (local):
         {'class_name': true class, 'pred_class': classifier's class (or None if unknown),
          'cls_conf': float, 'unknown': bool, 'confidence': segmentation confidence,
          'attributes': {...}, 'diag': {...} or None}
    t1: class_name -> T1 description sentence.

    Reports
      classifier_accuracy_pct   top-1 class right, over all sampled images (unknown counted as wrong)
      unknown_n / unknown_pct   images rejected by the unknown-class check (no report generated)
      review_flagged_n / _pct   reports whose segmentation confidence < review_threshold
      among generated reports (n_reports):
        structural_validity_pct, grounding_rate_pct, faithfulness_pct  (checks above; grounding is
            against the cause of the class the LLM was GIVEN, i.e. the predicted class)
        correct_cause_pct        predicted class == true class, so the cause in the report is the
                                 right one for this defect (causes are looked up per class)
    """
    n = len(records)
    if n == 0:
        return {}
    unknown = [r for r in records if r.get("unknown")]
    reports = [r for r in records if not r.get("unknown") and r.get("diag") is not None]
    right = sum(1 for r in records if not r.get("unknown") and r.get("pred_class") == r["class_name"])
    flagged = sum(1 for r in reports if r.get("confidence", 1.0) < review_threshold)
    out = {
        "n_sampled": n,
        "classifier_accuracy_pct": round(100 * right / n, 1),
        "unknown_n": len(unknown),
        "unknown_pct": round(100 * len(unknown) / n, 1),
        "n_reports": len(reports),
        "review_threshold": review_threshold,
        "review_flagged_n": flagged,
        "review_flagged_pct": round(100 * flagged / len(reports), 1) if reports else None,
        "llm_input": "predicted class from the ResNet34 classifier",
    }
    if reports:
        m = evaluate_batch([{"diag": r["diag"], "t1_cause": t1[r["pred_class"]],
                             "attributes": r["attributes"]} for r in reports])
        out.update({k: v for k, v in m.items() if k != "n"})
        out["correct_cause_pct"] = round(
            100 * sum(r["pred_class"] == r["class_name"] for r in reports) / len(reports), 1)
    return out


def per_class_pipeline(records: list[dict], t1: dict) -> dict:
    """Same checks, grouped by the TRUE class of each sampled image."""
    from collections import defaultdict
    pc = defaultdict(lambda: {"n": 0, "unknown": 0, "class_right": 0, "reports": 0,
                              "valid": 0, "ground": 0, "faith": 0})
    for r in records:
        s = pc[r["class_name"]]
        s["n"] += 1
        if r.get("unknown"):
            s["unknown"] += 1
            continue
        s["class_right"] += int(r.get("pred_class") == r["class_name"])
        d = r.get("diag")
        if d is None:
            continue
        s["reports"] += 1
        s["valid"] += int(check_structural_validity(d))
        s["ground"] += int(check_grounding(d, t1[r["pred_class"]]))
        s["faith"] += int(check_faithfulness(d, r["attributes"]))
    return {c: dict(v) for c, v in pc.items()}
