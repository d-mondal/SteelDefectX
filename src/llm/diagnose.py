"""
LLM diagnostic layer.

Turns a PREDICTION (deterministic T3-style attributes + class + confidence) into a
DIAGNOSTIC note: likely root cause + severity + recommended inspection action.

Design guards against hallucination:
  - The LLM never sees the raw image. It only sees FACTS we computed (attributes) plus the
    class's real industrial cause from the paper's T1 vocabulary.
  - It is instructed to ground its cause in the provided T1 cause, not invent one.
  - Output is required as strict JSON so it's parseable + checkable.

The API call is isolated in `_call_llm` so the provider can be swapped in one place.
"""
from __future__ import annotations
import json
import os
import re


SYSTEM_PROMPT = """You are a steel manufacturing quality-control assistant. You convert \
structured defect measurements into a concise DIAGNOSTIC note for a shop-floor inspector.

Rules you must follow:
- Base your 'likely_cause' on the PROVIDED industrial cause for this defect class. Do not \
invent a different physical cause. You may rephrase or add mechanism detail consistent with it.
- Base 'severity' on the measured scale and saliency provided (larger + higher saliency = more severe).
- 'recommended_action' must be a concrete inspection/handling step (e.g. which process stage to check, \
whether to hold the coil, whether human review is needed).
- If confidence is low, explicitly recommend human verification.
- Be specific and terse. Never describe the image; reason only from the measurements given.
- Respond ONLY with a JSON object, no prose outside it, with keys exactly: \
likely_cause, severity, recommended_action, summary. \
'severity' must be one of: low, moderate, high. 'summary' is one plain-language sentence."""


def build_user_prompt(defect_class: str, t1_cause: str, attributes: dict,
                      confidence: float | None, uncertainty: float | None) -> str:
    attrs = {k: v for k, v in attributes.items() if k != "_raw"}
    conf_line = ""
    if confidence is not None:
        conf_line = f"\nModel confidence: {confidence:.2f}"
        if uncertainty is not None:
            conf_line += f" | mean predictive uncertainty: {uncertainty:.3f}"
        conf_line += ("\n(NOTE: confidence is low — recommend human verification.)"
                      if confidence < 0.6 else "")
    return f"""Defect class: {defect_class}
Known industrial cause for this class (ground your likely_cause in this): "{t1_cause}"

Deterministic measurements of THIS instance:
{json.dumps(attrs, indent=2)}{conf_line}

Produce the diagnostic JSON now."""


def _call_llm(system: str, user: str, provider: str = "gemini",
              model: str | None = None, max_tokens: int = 1024) -> str:
    """Isolated provider call. Swap provider here without touching the rest of the layer.
    Reads the API key from env. Returns the raw text response."""
    provider = provider.lower()
    if provider == "gemini":
        # Current Google GenAI SDK (google-genai). Stable replacement for the
        # deprecated google.generativeai. Key from GEMINI_API_KEY.
        import time as _time
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        last_err = None
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model or "gemini-2.5-flash",
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        max_output_tokens=max_tokens,
                        temperature=0.3,
                        response_mime_type="application/json",
                    ),
                )
                return resp.text
            except Exception as e:
                last_err = e
                _time.sleep(2 * (attempt + 1))
        raise last_err
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=model or "claude-sonnet-4-5",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content
    raise ValueError(f"unknown provider {provider}")


def _parse_json(text: str) -> dict:
    """Extract the JSON object from the model response, robust to markdown fences,
    ```json wrappers, and leading/trailing prose."""
    if not text:
        raise ValueError("empty response")
    t = text.strip()
    # strip a leading ```json or ``` fence and any trailing ``` fence
    t = re.sub(r"^\s*```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    t = t.strip()
    # grab the outermost {...} block
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in response: {text[:200]}")
    return json.loads(t[start:end + 1])


def diagnose(defect_class: str, t1_cause: str, attributes: dict,
             confidence: float | None = None, uncertainty: float | None = None,
             provider: str = "gemini", model: str | None = None) -> dict:
    """Full diagnostic call. Returns dict with keys:
       likely_cause, severity, recommended_action, summary, plus _valid (bool) and _raw_response."""
    user = build_user_prompt(defect_class, t1_cause, attributes, confidence, uncertainty)
    raw = _call_llm(SYSTEM_PROMPT, user, provider=provider, model=model)
    try:
        out = _parse_json(raw)
        required = {"likely_cause", "severity", "recommended_action", "summary"}
        out["_valid"] = required.issubset(out.keys()) and out.get("severity") in {"low", "moderate", "high"}
    except Exception as e:
        out = {"likely_cause": None, "severity": None, "recommended_action": None,
               "summary": None, "_valid": False, "_parse_error": str(e)}
    out["_raw_response"] = raw
    return out


def load_t1_causes(class_descriptions_path: str) -> dict:
    """class_name -> the T1 description sentence (which contains the industrial cause)."""
    with open(class_descriptions_path, encoding="utf-8") as f:
        return json.load(f)