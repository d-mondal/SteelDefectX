"""
SteelDefectX — Explainable Defect Inspection Dashboard (Streamlit)

One screen, sized so a full result fits in a single screenshot:
  sidebar:  upload image(s) -> Run inspection (and pick which result to show if several)
  main:     status row  (class + unknown-class check, severity, segmentation confidence + review flag)
            images      (input | predicted mask on the image | MC-Dropout uncertainty)
            report row  (measured attributes | likely cause | recommended action | summary)

LLM mode: tries a LIVE call; if the key is missing / quota is hit / it errors, it falls back
to a cached diagnosis from the eval run so the demo never breaks in front of an audience.
Evaluation results live in results/ and in the report; the old Analytics tab was removed.

Run:  streamlit run dashboard/app.py
Needs:  models/unet_dropout.pt, models/classifier.pt + class_names.json + class_threshold.json,
        results/llm/diagnoses.jsonl (cached fallback), and (for live LLM) GEMINI_API_KEY in env.
"""
import os
import sys
import json
import html
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.express as px

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# make 'src' importable when run from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import REVIEW_CONFIDENCE_THRESHOLD, UNKNOWN_CLASS_THRESHOLD_FILE
try:
    from dotenv import load_dotenv as _ld
    _ld(ROOT / ".env")
except Exception:
    pass

st.set_page_config(page_title="SteelDefectX Inspector", layout="wide", initial_sidebar_state="expanded")

CKPT = str(ROOT / "models" / "unet_dropout.pt")
CLS_CKPT = str(ROOT / "models" / "classifier.pt")
CLS_NAMES = str(ROOT / "models" / "class_names.json")
CLS_THRESHOLD = str(ROOT / UNKNOWN_CLASS_THRESHOLD_FILE)
RESULTS = ROOT / "results"
LLM_MODEL = "gemini-3.5-flash-lite"   # model used for the eval run
MC_PASSES = 10                        # dashboard passes (the evaluation used 30)
IMG_HEIGHT = 280                     # px per image panel; lower it if your screen is short

# ------------------------- UI styling -------------------------
st.html("""
<style>
  [data-testid="stHeader"] { background: transparent; }
  .block-container, [data-testid="stMainBlockContainer"] {
      padding-top: 2.6rem !important; padding-bottom: 0.6rem !important; max-width: 100% !important;
  }
  [data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] { padding-top: 1.2rem !important; }
  .st-key-run_inspect button { font-size: 1.05rem !important; font-weight: 650 !important; min-height: 2.8rem !important; }

  .sdx-title { font-size: 1.45rem; font-weight: 750; margin: 0 0 0.45rem 0; line-height: 1.25; }
  .sdx-title small { font-size: 0.95rem; font-weight: 500; color: #6e7781; }
  .sdx-pills { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin: 0 0 0.5rem 0; }
  .sdx-pill { padding: 0.32rem 0.85rem; border-radius: 999px; font-weight: 650; font-size: 0.95rem;
              border: 1.5px solid; white-space: nowrap; }
  .sdx-grid { display: grid; grid-template-columns: 1fr 1.15fr 1.15fr 1fr; gap: 0.75rem; margin-top: 0.15rem; }
  .sdx-card { border: 1px solid #d9dee5; border-radius: 0.65rem; padding: 0.75rem 0.95rem;
              background: #ffffff; color: #1f2328; }
  .sdx-card h4 { margin: 0 0 0.4rem 0; font-size: 0.95rem; font-weight: 700; color: inherit; }
  .sdx-card p { margin: 0; font-size: 0.93rem; line-height: 1.45; }
  .sdx-card .sdx-muted { color: #6e7781; font-size: 0.8rem; line-height: 1.35; margin-top: 0.4rem; }
  .sdx-attr { display: grid; grid-template-columns: auto 1fr; column-gap: 0.7rem; row-gap: 0.12rem; font-size: 0.88rem; }
  .sdx-attr .k { color: #6e7781; }
  .sdx-wide { grid-column: span 3; }
  .sdx-warn { border-color: #bf8700; background: #fff8e6; }
</style>
""")

DARK_CSS = """
<style>
  [data-testid="stAppViewContainer"], [data-testid="stBottomBlockContainer"] { background-color: #0d1117 !important; }
  [data-testid="stHeader"] { background: transparent !important; }
  [data-testid="stSidebar"] { background-color: #161b22 !important; }
  [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] h3, [data-testid="stCaptionContainer"] p,
  [data-testid="stWidgetLabel"] p, .sdx-title { color: #e6edf3 !important; }
  .sdx-card { background: #161b22 !important; color: #e6edf3 !important; border-color: #30363d !important; }
  .sdx-warn { background: #2b2111 !important; border-color: #bf8700 !important; }
  [data-testid="stFileUploader"] section { background-color: #0d1117 !important; border-color: #30363d !important; }
</style>
"""


# ------------------------- cached loaders -------------------------
@st.cache_resource
def load_model():
    try:
        import torch
        from src.segmentation.model import build_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model("unet", "resnet34", None, dropout=0.2)
        model.load_state_dict(torch.load(CKPT, map_location=device))
        model.to(device).eval()
        return model, device
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_classifier():
    """Returns (net, class_names, device, unknown_threshold); net is None if not trained yet."""
    if not (os.path.exists(CLS_CKPT) and os.path.exists(CLS_NAMES)):
        return None, None, None, None
    try:
        import torch
        from src.segmentation.classifier import build_classifier, load_unknown_threshold
        device = "cuda" if torch.cuda.is_available() else "cpu"
        names = json.load(open(CLS_NAMES))
        net = build_classifier(len(names), pretrained=False)
        net.load_state_dict(torch.load(CLS_CKPT, map_location=device))
        net.to(device).eval()
        return net, names, device, load_unknown_threshold(CLS_THRESHOLD)
    except Exception:
        return None, None, None, None


@st.cache_data
def load_t1():
    p = ROOT / "dashboard" / "class_descriptions.json"
    if p.exists():
        return json.load(open(p, encoding="utf-8"))
    from huggingface_hub import hf_hub_download
    fp = hf_hub_download("Zhaosxian/SteelDefectX", "class_descriptions.json", repo_type="dataset")
    return json.load(open(fp, encoding="utf-8"))


@st.cache_data
def load_cached_diagnoses():
    d = RESULTS / "llm" / "diagnoses.jsonl"
    return [json.loads(l) for l in open(d, encoding="utf-8")] if d.exists() else []


# ------------------------- inference helpers -------------------------
MEAN = np.array([0.485, 0.456, 0.406])[None, :, None, None]
STD = np.array([0.229, 0.224, 0.225])[None, :, None, None]


def predict(model, device, gray, mc=MC_PASSES):
    import torch
    from src.segmentation.model import enable_mc_dropout
    x = np.stack([gray, gray, gray], 0)[None].astype(np.float32) / 255.0
    x = torch.tensor((x - MEAN) / STD, dtype=torch.float32).to(device)
    enable_mc_dropout(model)
    with torch.no_grad():
        ps = torch.stack([torch.sigmoid(model(x)) for _ in range(mc)], 0)
    prob = ps.mean(0)[0, 0].cpu().numpy()
    unc = ps.std(0)[0, 0].cpu().numpy()
    conf = float(prob[prob > 0.5].mean()) if (prob > 0.5).any() else float(1 - prob.mean())
    return prob, unc, conf


def try_live_llm(cls, t1_cause, attrs, conf, unc_mean):
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("no GEMINI_API_KEY (add it to .env at the repo root)")
    from src.llm.diagnose import diagnose
    d = diagnose(cls, t1_cause, attrs, confidence=conf, uncertainty=unc_mean,
                 provider="gemini", model=LLM_MODEL)
    if not d.get("_valid"):
        raise RuntimeError(f"live response invalid: {d.get('_parse_error', 'missing fields')}")
    return d


def short_reason(e):
    """One-line reason for the fallback, shown on the LLM pill (full text on hover)."""
    msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    low = msg.lower()
    if "gemini_api_key" in low:
        return "no API key"
    if "429" in msg or "quota" in low or "exhausted" in low:
        return "quota / rate limit"
    if "403" in msg or "401" in msg or "api key not valid" in low or "permission" in low:
        return "key rejected"
    if "404" in msg or "not found" in low:
        return "model not found"
    if "connect" in low or "timeout" in low or "resolve" in low or "network" in low:
        return "no internet"
    return type(e).__name__


def cached_llm_for_class(cls, diagnoses, image_name=None):
    """Fallback: this image's own cached report if it was in the eval run, else one of the same class."""
    for r in diagnoses:
        if image_name and r.get("image_name") == image_name and (r.get("diag") or {}).get("_valid"):
            return r["diag"]
    for r in diagnoses:
        if r.get("pred_class", r.get("class_name")) == cls and (r.get("diag") or {}).get("_valid"):
            return r["diag"]
    for r in diagnoses:
        if (r.get("diag") or {}).get("_valid"):
            return r["diag"]
    return None


# ------------------------- display helpers -------------------------
SEVERITY_COLOR = {"high": "#cf222e", "moderate": "#bf8700", "low": "#1a7f37"}
SEVERITY_EMOJI = {"high": "🔴", "moderate": "🟠", "low": "🟢"}


def confidence_color(c):
    if c is None:
        return "#57606a"
    if c >= REVIEW_CONFIDENCE_THRESHOLD:
        return "#1a7f37"
    if c >= 0.6:
        return "#bf8700"
    return "#cf222e"


def pill(text, color):
    return (f'<span class="sdx-pill" style="color:{color};border-color:{color};'
            f'background:{color}1a">{text}</span>')


def esc(x):
    return html.escape(str(x)) if x is not None else "—"


def show_image(img, title, dark):
    fig = px.imshow(img, binary_string=img.ndim == 2, aspect="equal")
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=0.985, yanchor="top",
                   font=dict(size=14, color="#e6edf3" if dark else "#57606a")),
        margin=dict(l=0, r=0, t=26, b=0), height=IMG_HEIGHT,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True, "displayModeBar": "hover",
                            "modeBarButtonsToRemove": ["select2d", "lasso2d"]})


def uncertainty_rgb(unc):
    """Colour the std map with inferno (black = sure, yellow = unsure), scaled to this image's max."""
    u = unc / max(float(unc.max()), 1e-6)
    try:
        from matplotlib import colormaps
        return (colormaps["inferno"](u)[..., :3] * 255).astype(np.uint8)
    except Exception:
        g = (u * 255).astype(np.uint8)
        return np.stack([g] * 3, -1)


def mask_overlay(gray, mask):
    rgb = np.stack([gray] * 3, -1).astype(np.float32)
    m = mask > 0
    rgb[m] = 0.45 * rgb[m] + 0.55 * np.array([220.0, 38.0, 38.0])
    return rgb.clip(0, 255).astype(np.uint8)


def attr_rows(attrs):
    pos = attrs.get("Position") or []
    pos_names = [p[0] if isinstance(p, (list, tuple)) else str(p) for p in pos][:2]
    rows = [("Scale", attrs.get("Scale")), ("Polarity", attrs.get("Polarity")),
            ("Saliency", attrs.get("Saliency")), ("Defects", attrs.get("Number of Defects")),
            ("Spread", attrs.get("Spatial Distribution")), ("Shape", attrs.get("Shape")),
            ("Direction", attrs.get("Direction")), ("Position", ", ".join(pos_names) or "—")]
    return "".join(f'<span class="k">{k}</span><span>{esc(v)}</span>' for k, v in rows)


def render(r, dark, unknown_thr):
    is_unknown, cls, cls_conf, conf, diag = r["is_unknown"], r["cls"], r["cls_conf"], r["conf"], r["diag"]

    # ---- title + status pills ----
    pills = []
    if is_unknown:
        pills.append(pill(f"❓ Unknown class: requires manual class labelling (p = {cls_conf:.2f})", "#cf222e"))
    else:
        p_txt = f" · p = {cls_conf:.2f}" if cls_conf is not None else " · manual"
        pills.append(pill(f"🔧 {esc(cls)}{p_txt}", confidence_color(cls_conf)))
    if diag:
        sev = (diag.get("severity") or "").lower()
        pills.append(pill(f"{SEVERITY_EMOJI.get(sev, '⚪')} Severity: {esc(sev or 'n/a')}", SEVERITY_COLOR.get(sev, "#57606a")))
    pills.append(pill(f"Segmentation confidence {conf:.2f}", confidence_color(conf)))
    if conf < REVIEW_CONFIDENCE_THRESHOLD:
        pills.append(pill(f"⚠️ Flagged for manual review (below {REVIEW_CONFIDENCE_THRESHOLD:.0%})", confidence_color(conf)))
    if r["source"] == "live":
        pills.append(pill("LLM: live", "#1a7f37"))
    elif r["source"] == "cached":
        why = esc(r.get("llm_error") or "")
        pills.append(pill(f'<span title="{why}">LLM: cached · {esc(r.get("llm_reason"))}</span>', "#57606a"))
    st.html(f'<div class="sdx-title">🔩 SteelDefectX: Explainable Defect Inspector '
            f'<small>· {esc(r["file_name"])}</small></div><div class="sdx-pills">{"".join(pills)}</div>')

    # ---- three images ----
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        show_image(r["gray"], "Input", dark)
    with c2:
        show_image(mask_overlay(r["gray"], r["mask"]), "Predicted defect (red)", dark)
    with c3:
        show_image(uncertainty_rgb(r["unc"]), f"Uncertainty (MC-Dropout std, max {r['unc'].max():.2f})", dark)

    # ---- report row: one HTML grid so all four cards line up ----
    attrs_card = f'<div class="sdx-card"><h4>Measured attributes</h4><div class="sdx-attr">{attr_rows(r["attrs"])}</div></div>'
    if diag:
        cause_note = (f'<p class="sdx-muted">Dataset cause for this class: {esc(r["t1_cause"])}</p>'
                      if r.get("t1_cause") else "")
        cards = (f'<div class="sdx-card"><h4>Likely cause</h4><p>{esc(diag.get("likely_cause"))}</p>{cause_note}</div>'
                 f'<div class="sdx-card"><h4>Recommended action</h4><p>{esc(diag.get("recommended_action"))}</p></div>'
                 f'<div class="sdx-card"><h4>Summary</h4><p>{esc(diag.get("summary"))}</p></div>')
    elif is_unknown:
        thr = f"{unknown_thr:.2f}" if unknown_thr is not None else "the threshold"
        cards = (f'<div class="sdx-card sdx-wide sdx-warn"><h4>No report: unknown class</h4><p>'
                 f'Classifier confidence {cls_conf:.2f} is below the unknown-class threshold {thr}, so this defect '
                 'does not match any of the 25 known classes with enough confidence. It needs manual class '
                 'labelling; no cause or diagnostic note is generated. The mask, uncertainty map and measured '
                 'attributes are still shown.</p></div>')
    else:
        cards = ('<div class="sdx-card sdx-wide sdx-warn"><h4>No diagnosis available</h4>'
                 '<p>No live LLM key and no cached example for this class.</p></div>')
    st.html(f'<div class="sdx-grid">{attrs_card}{cards}</div>')


# ------------------------- sidebar -------------------------
net_cls, cls_names, cls_device, unknown_thr = load_classifier()

with st.sidebar:
    st.markdown("### 🔩 SteelDefectX")
    up = st.file_uploader("Steel surface image(s)", type=["png", "jpg", "jpeg", "bmp"],
                          key="uploader", accept_multiple_files=True)
    manual_class = None
    if net_cls is None:
        st.caption("ℹ️ Classifier not found (train it in notebook 05); pick the class manually.")
        manual_class = st.selectbox("Defect class (manual)", options=sorted(load_t1().keys()))
    elif unknown_thr is None:
        st.caption("ℹ️ Unknown-class threshold not calibrated yet, so no class is rejected.")
    run_inspection = st.button("Run inspection", type="primary", disabled=not up,
                               key="run_inspect", use_container_width=True)
    picker_slot = st.empty()
    st.divider()
    dark_mode = st.toggle("🌙 Dark", key="dark_mode")
    st.caption("Gemini key found: live reports" if os.environ.get("GEMINI_API_KEY")
               else "⚠️ No GEMINI_API_KEY: reports will be cached")
    thr_txt = f"{unknown_thr:.2f}" if unknown_thr is not None else "not set"
    st.caption(f"Unknown-class threshold {thr_txt} · review threshold "
               f"{REVIEW_CONFIDENCE_THRESHOLD:.0%} · {MC_PASSES} MC-Dropout passes")

if dark_mode:
    st.html(DARK_CSS)

if "inspection_results" not in st.session_state:
    st.session_state.inspection_results = {}

# ------------------------- run inspection -------------------------
if run_inspection and up:
    import cv2
    from src.llm.attributes import extract_attributes
    model, device = load_model()
    if model is None:
        st.error(
            f"⚠️ **Could not load PyTorch segmentation model:**\n\n`{device}`\n\n"
            "**If Windows Smart App Control is enabled:** Windows blocks unsigned PyTorch binaries (`_C.pyd`). "
            "Open **Windows Security → App & browser control → Smart App Control settings** and set it to "
            "**Evaluation** or **Off**.")
        st.stop()

    st.session_state.inspection_results = {}
    t1 = load_t1()
    diagnoses = load_cached_diagnoses()
    for file in up:
        gray = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            st.error(f"Could not read {file.name}")
            continue
        gray = cv2.resize(gray, (256, 256))
        with st.spinner(f"Inspecting {file.name}..."):
            prob, unc, conf = predict(model, device, gray)
            attrs = extract_attributes(gray, prob)
            is_unknown, cls_conf = False, None
            if net_cls is not None:
                from src.segmentation.classifier import predict_class, apply_unknown_threshold
                top_cls, cls_conf, _ = predict_class(net_cls, gray, cls_names, cls_device)
                cls, is_unknown = apply_unknown_threshold(top_cls, cls_conf, unknown_thr)
            else:
                cls = manual_class
            diag, source, llm_reason, llm_error = None, None, None, None
            if is_unknown:
                source = "skipped"   # no class -> no dataset cause -> no LLM call
            else:
                try:
                    diag = try_live_llm(cls, t1[cls], attrs, conf, float(unc.mean()))
                    source = "live"
                except Exception as e:
                    llm_reason, llm_error = short_reason(e), f"{type(e).__name__}: {e}"[:400]
                    print(f"[SteelDefectX] live LLM failed for {file.name}: {llm_error}")
                    diag = cached_llm_for_class(cls, diagnoses, file.name)
                    source = "cached" if diag else None
        st.session_state.inspection_results[file.name] = {
            "file_name": file.name, "gray": gray, "prob": prob, "mask": (prob > 0.5).astype(np.uint8) * 255,
            "unc": unc, "conf": conf, "attrs": attrs, "cls": cls, "cls_conf": cls_conf,
            "is_unknown": is_unknown, "diag": diag, "source": source,
            "llm_reason": llm_reason, "llm_error": llm_error,
            "t1_cause": None if is_unknown else t1.get(cls),
        }

# ------------------------- main area -------------------------
results = st.session_state.inspection_results
if not results:
    st.html('<div class="sdx-title">🔩 SteelDefectX: Explainable Defect Inspector</div>')
    st.info("Upload a steel surface image in the sidebar and press **Run inspection**. "
            "The class, mask, uncertainty, measured attributes and the diagnostic report appear on one screen.")
else:
    names = list(results)
    shown = names[0]
    if len(names) > 1:
        shown = picker_slot.selectbox("Show result for", names, key="shown_file")
    render(results[shown], dark_mode, unknown_thr)
