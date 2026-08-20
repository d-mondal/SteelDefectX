"""
SteelDefectX — Explainable Defect Inspection Dashboard (Streamlit)

Ties the whole system into one demo:
  Tab 1  Inspect: upload an image -> segmentation mask + MC-Dropout uncertainty heatmap
                  -> deterministic T3 attributes -> LLM diagnostic note -> human-review flag.
  Tab 2  Analytics: segmentation head-to-head, per-class IoU, calibration summary.
  Tab 3  Eval results: browse the LLM diagnostic evaluation + example notes.

LLM mode: tries a LIVE call; if the key is missing / quota is hit / it errors, it falls back
to a cached diagnosis from the eval run so the demo never breaks in front of an audience.

Run:  streamlit run dashboard/app.py
Needs:  models/unet_dropout.pt, results/*, and (for live LLM) GEMINI_API_KEY in env.
"""
import os
import sys
import json
import glob
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# make 'src' importable when run from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv as _ld
    _ld(ROOT / ".env")
except Exception:
    pass

st.set_page_config(page_title="SteelDefectX Inspector", layout="wide")

# ------------------------- UI styling -------------------------
st.html("""
<style>

    /* ---------- Main title ---------- */
    .main-title {
        text-align: center;
        font-size: 2.7rem;
        font-weight: 750;
        margin-top: 0.5rem;
        margin-bottom: 2rem;
    }

    /* ---------- Sidebar navigation ---------- */
    [data-testid="stSidebar"] {
        padding-top: 1rem;
    }

    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.35rem;
    }

    [data-testid="stSidebar"] .stRadio label {
        padding: 0.65rem 0.8rem;
        border-radius: 0.6rem;
        font-size: 1.05rem;
        font-weight: 600;
    }

    /* ---------- Upload heading ---------- */
    .upload-heading {
        text-align: center;
        font-size: 1.7rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 1.2rem;
    }

    /* ---------- Upload box ---------- */
    /* ---------- Responsive upload section ---------- */

    .st-key-uploader {
        display: flex;
        justify-content: center;
        width: 100%;
    }

    .st-key-uploader [data-testid="stFileUploaderDropzone"] {
        width: min(60vw, 650px) !important;
        margin: 0 auto !important;
        box-sizing: border-box !important;

        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;

        padding: 1rem !important;
    }

    /* Upload button */
    .st-key-uploader [data-testid="stFileUploaderDropzone"] button {
        font-size: 1.05rem !important;
        padding: 0.65rem 1.5rem !important;
        margin: 0 auto !important;
    }

    /* Helper text */
    .st-key-uploader [data-testid="stFileUploaderDropzone"] small {
        text-align: center !important;
        margin-top: 0.4rem !important;
    }

    /* ---------- Run inspection button ---------- */
    .st-key-run_inspect button {
        font-size: 1.15rem !important;
        font-weight: 650 !important;
        padding: 0.75rem 2rem !important;
        min-height: 3rem !important;
        border-radius: 0.6rem !important;
    }

    /* ---------- Confidence ---------- */
    .confidence-box {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.75rem 1.5rem;
        border-radius: 0.6rem;
        font-size: 1.15rem;
        font-weight: 650;
        margin: 1.5rem 0;
    }

    /* ---------- Class / severity ---------- */
    .result-header {
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 1rem;
    }

    /* ---------- Diagnostic cards ---------- */
    .diagnostic-card {
        min-height: 180px;
        height: 180px;
        padding: 1rem 1.1rem;
        border: 1px solid #d9dee5;
        border-radius: 0.65rem;
        box-sizing: border-box;
        overflow: hidden;
    }

    .diagnostic-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }

    .diagnostic-card-text {
        font-size: 1rem;
        line-height: 1.55;
    }

    /* ---------- T3 Attributes ---------- */

    .attributes-heading {
        font-size: 1.15rem;
        font-weight: 700;
    }

    /* ---------- T3 dropdown styled like a button ---------- */

    [class*="st-key-t3_block_"] [data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
    }

    [class*="st-key-t3_block_"] [data-testid="stExpander"] summary {
        background-color: #2563EB !important;
        color: white !important;
        border: 1px solid #2563EB !important;
        border-radius: 0.6rem !important;

        font-size: 1.05rem !important;
        font-weight: 650 !important;

        padding: 0.65rem 1.4rem !important;
        min-height: 3rem !important;

        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12) !important;

        width: fit-content !important;
        margin: 0 auto !important;
    }

    [class*="st-key-t3_block_"] [data-testid="stExpander"] summary p {
        color: white !important;
    }

    [class*="st-key-t3_block_"] [data-testid="stExpander"] summary:hover {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
    }

    [class*="st-key-t3_block_"] [data-testid="stExpander"] details {
        border: none !important;
    }

    /* T3 content box */
    .t3-content {
        margin-top: 0.8rem;
        margin-bottom: 1rem;
        padding: 1rem 1.2rem;

        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 0.65rem;
    }

    </style>
""")

CKPT = str(ROOT / "models" / "unet_dropout.pt")
CLS_CKPT = str(ROOT / "models" / "classifier.pt")
CLS_NAMES = str(ROOT / "models" / "class_names.json")
RESULTS = ROOT / "results"
LLM_MODEL = "gemini-3.5-flash-lite"   # model used for the eval run


# ------------------------- cached loaders -------------------------
@st.cache_resource
def load_model():
    import torch
    from src.segmentation.model import build_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("unet", "resnet34", None, dropout=0.2)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.to(device).eval()
    return model, device


@st.cache_resource
def load_classifier():
    """Returns (net, class_names, device) or (None, None, None) if not trained yet."""
    if not (os.path.exists(CLS_CKPT) and os.path.exists(CLS_NAMES)):
        return None, None, None
    import torch
    from src.segmentation.classifier import build_classifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = json.load(open(CLS_NAMES))
    net = build_classifier(len(names), pretrained=False)
    net.load_state_dict(torch.load(CLS_CKPT, map_location=device))
    net.to(device).eval()
    return net, names, device


@st.cache_data
def load_t1():
    # bundled small copy or fetch from HF once
    p = ROOT / "dashboard" / "class_descriptions.json"
    if p.exists():
        return json.load(open(p, encoding="utf-8"))
    from huggingface_hub import hf_hub_download
    fp = hf_hub_download("Zhaosxian/SteelDefectX", "class_descriptions.json", repo_type="dataset")
    return json.load(open(fp, encoding="utf-8"))


@st.cache_data
def load_seg_summary():
    f = RESULTS / "segmentation" / "comparison_summary.json"
    return json.load(open(f)) if f.exists() else None


@st.cache_data
def load_per_class():
    f = RESULTS / "segmentation" / "unet_resnet34_seed0_test_metrics.json"
    return json.load(open(f))["per_class"] if f.exists() else None


@st.cache_data
def load_calibration():
    f = RESULTS / "uncertainty" / "calibration.json"
    return json.load(open(f)) if f.exists() else None


@st.cache_data
def load_llm_eval():
    m = RESULTS / "llm" / "eval_metrics.json"
    d = RESULTS / "llm" / "diagnoses.jsonl"
    metrics = json.load(open(m)) if m.exists() else None
    diagnoses = [json.loads(l) for l in open(d)] if d.exists() else []
    return metrics, diagnoses


# ------------------------- inference helpers -------------------------
MEAN = np.array([0.485, 0.456, 0.406])[None, :, None, None]
STD = np.array([0.229, 0.224, 0.225])[None, :, None, None]


def predict(model, device, gray, mc=10):
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
    """Live diagnosis; returns (diag, source) where source is 'live' or raises."""
    from src.llm.diagnose import diagnose
    d = diagnose(cls, t1_cause, attrs, confidence=conf, uncertainty=unc_mean,
                 provider="gemini", model=LLM_MODEL)
    if not d.get("_valid"):
        raise RuntimeError("live response invalid")
    return d


def cached_llm_for_class(cls, diagnoses):
    """Fallback: a cached diagnosis of the same class from the eval run."""
    for r in diagnoses:
        if r.get("class_name") == cls and r.get("diag", {}).get("_valid"):
            return r["diag"]
    # otherwise any valid one
    for r in diagnoses:
        if r.get("diag", {}).get("_valid"):
            return r["diag"]
    return None


# ------------------------- confidence / severity color helpers -------------------------
SEVERITY_COLOR = {"high": "#cf222e", "moderate": "#bf8700", "low": "#1a7f37"}
SEVERITY_EMOJI = {"high": "🔴", "moderate": "🟠", "low": "🟢"}


def confidence_color(c):
    """3-tier color for a 0-1 confidence score."""
    if c is None:
        return "#57606a"
    if c >= 0.8:
        return "#1a7f37"
    if c >= 0.6:
        return "#bf8700"
    return "#cf222e"


def confidence_badge(label, conf):
    color = confidence_color(conf)

    st.html(f"""
    <div style="text-align:center; margin:1.5rem 0;">
        <div class="confidence-box"
             style="
                background-color:{color}1f;
                border:1.5px solid {color};
                color:{color};
             ">
            {label} - {conf:.2f}
        </div>
    </div>
    """)


def inject_dark_mode_css():
    """App-wide dark palette override. No-op in light mode (default Streamlit theme)."""
    bg, bg2, text, border = "#0d1117", "#161b22", "#e6edf3", "#30363d"
    st.html(f"""
    <style>
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stBottomBlockContainer"] {{
        background-color: {bg} !important;
    }}
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stCaptionContainer"] p,
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stJson"],
    [data-testid="stTabs"] button[role="tab"] p {{
        color: {text} !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stExpander"] details,
    [data-testid="stFileUploader"] section,
    [data-testid="stPopoverBody"] {{
        background-color: {bg2} !important;
        border-color: {border} !important;
    }}
    [data-testid="stBaseButton-secondary"] {{
        background-color: {bg2} !important;
        color: {text} !important;
        border-color: {border} !important;
    }}
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stTextInput"] input {{
        background-color: {bg2} !important;
        color: {text} !important;
        border-color: {border} !important;
    }}
    </style>
    """)


# ------------------------- UI -------------------------
# ------------------------- HEADER -------------------------

st.markdown(
    '<div class="main-title">🔩 SteelDefectX — Explainable Defect Inspector</div>',
    unsafe_allow_html=True
)

# ------------------------- SIDEBAR NAVIGATION -------------------------

with st.sidebar:

    st.markdown("### Navigation")

    page = st.radio(
        "Go to",
        [
            "🔍 Inspect",
            "📊 Analytics",
            "🧪 Eval results"
        ],
        label_visibility="collapsed",
        key="navigation"
    )

    st.divider()

    dark_mode = st.toggle("🌙 Dark", key="dark_mode")

if dark_mode:
    inject_dark_mode_css()

# ===== TAB 1: single-image pipeline =====
if page == "🔍 Inspect":
    import cv2
    st.markdown(
    '<div class="upload-heading">Upload a steel surface image</div>',
    unsafe_allow_html=True
    )
    with st.container(horizontal_alignment="center"):
        up = st.file_uploader(
            "PNG / JPG",
            type=["png", "jpg", "jpeg", "bmp"],
            key="uploader",
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

    net_cls, cls_names, cls_device = load_classifier()
    if net_cls is None:
        st.caption("ℹ️ Classifier not found — train it (notebook 05) to auto-assign the class. "
                   "You can still pick the class manually below.")
        manual_class = st.selectbox("Defect class (manual)", options=sorted(load_t1().keys()))
    else:
        manual_class = None

    st.write("")

    with st.container(horizontal_alignment="center"):
        run_inspection = st.button(
            "Run inspection",
            type="primary",
            disabled=not up,
            key="run_inspect"
        )

    st.write("")
    st.write("")

    # ------------------------------------------------------------
    # Persistent storage for inspection results
    # ------------------------------------------------------------
    if "inspection_results" not in st.session_state:
        st.session_state.inspection_results = {}


    # ============================================================
    # RUN INSPECTION
    # ============================================================
    if run_inspection and up:

        # Start a completely new inspection
        st.session_state.inspection_results = {}

        model, device = load_model()

        for i, file in enumerate(up):

            st.markdown("---")

            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    font-size:1.25rem;
                    font-weight:700;
                    margin:1.5rem 0 1rem 0;
                ">
                    🔍 Inspecting: {file.name}
                </div>
                """,
                unsafe_allow_html=True
            )

            # ----------------------------------------------------
            # Read image
            # ----------------------------------------------------
            file_bytes = np.frombuffer(file.read(), np.uint8)
            gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

            if gray is None:
                st.error(f"Could not read {file.name}")
                continue

            gray = cv2.resize(gray, (256, 256))

            # ----------------------------------------------------
            # Segmentation + uncertainty
            # ----------------------------------------------------
            with st.spinner(f"Inspecting {file.name}..."):
                prob, unc, conf = predict(model, device, gray)

            mask = (prob > 0.5).astype(np.uint8) * 255

            un = (unc - unc.min()) / (np.ptp(unc) + 1e-8)
            unc_display = (un * 255).astype(np.uint8)

            # ----------------------------------------------------
            # Deterministic T3 attributes
            #
            # IMPORTANT:
            # Compute them NOW, during inspection.
            # They are NOT computed when the T3 button is clicked.
            # ----------------------------------------------------
            from src.llm.attributes import extract_attributes

            attrs = extract_attributes(gray, prob)

            # ----------------------------------------------------
            # Classify
            # ----------------------------------------------------
            t1 = load_t1()

            if net_cls is not None:
                from src.segmentation.classifier import predict_class

                cls, cls_conf, _ = predict_class(
                    net_cls,
                    gray,
                    cls_names,
                    cls_device
                )
            else:
                cls = manual_class
                cls_conf = None

            # ----------------------------------------------------
            # LLM diagnosis
            #
            # Also computed ONLY during Run inspection.
            # ----------------------------------------------------
            _, diagnoses = load_llm_eval()

            diag, source = None, None

            try:
                diag = try_live_llm(
                    cls,
                    t1[cls],
                    attrs,
                    conf,
                    float(unc.mean())
                )
                source = "live"

            except Exception:
                diag = cached_llm_for_class(cls, diagnoses)
                source = "cached (live unavailable)"

            # ----------------------------------------------------
            # Save EVERYTHING in session state
            #
            # This is the important part.
            # T3 button will use this stored result.
            # ----------------------------------------------------
            st.session_state.inspection_results[file.name] = {
                "file_name": file.name,
                "gray": gray,
                "prob": prob,
                "mask": mask,
                "unc": unc,
                "unc_display": unc_display,
                "conf": conf,
                "attrs": attrs,
                "cls": cls,
                "cls_conf": cls_conf,
                "diag": diag,
                "source": source,
            }

            # ====================================================
            # DISPLAY INSPECTION RESULTS
            # ====================================================

            # ----------------------------------------------------
            # Zoomable image helper
            # ----------------------------------------------------
            def show_zoomable_image(image, title):

                fig = px.imshow(
                    image,
                    binary_string=True,
                    aspect="equal"
                )

                fig.update_layout(
                    title=None,

                    margin=dict(
                        l=0,
                        r=0,
                        t=0,
                        b=35
                    ),

                    height=315,

                    xaxis=dict(
                        visible=False,
                        showgrid=False,
                        zeroline=False,
                    ),

                    yaxis=dict(
                        visible=False,
                        showgrid=False,
                        zeroline=False,
                        scaleanchor="x",
                        scaleratio=1,
                    ),
                )

                fig.update_xaxes(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False
                )

                fig.update_yaxes(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    scaleanchor="x",
                    scaleratio=1
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displaylogo": False,
                        "scrollZoom": True,
                        "displayModeBar": "hover",
                        "modeBarButtonsToRemove": [
                            "select2d",
                            "lasso2d"
                        ]
                    }
                )

                st.markdown(
                    f"""
                    <div style="
                        text-align:center;
                        color:#8a8f98;
                        font-size:0.9rem;
                        margin-top:-1.8rem;
                        margin-bottom:0.8rem;
                    ">
                        {title}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ----------------------------------------------------
            # Three images
            # ----------------------------------------------------
            c1, c2, c3 = st.columns(
                [1, 1, 1],
                gap="small"
            )

            with c1:
                show_zoomable_image(
                    gray,
                    "Input"
                )

            with c2:
                show_zoomable_image(
                    mask,
                    "Predicted mask"
                )

            with c3:
                show_zoomable_image(
                    unc_display,
                    "Uncertainty (MC-Dropout std)"
                )

            # ----------------------------------------------------
            # Model confidence
            # ----------------------------------------------------
            low_conf = conf < 0.6

            with st.container(horizontal_alignment="center"):

                confidence_badge(
                    "Model confidence",
                    conf
                )

                if low_conf:
                    st.markdown(
                        f"""
                        <div style="
                            text-align:center;
                            font-weight:600;
                            color:{confidence_color(conf)};
                        ">
                            ⚠️ Low confidence — human review advised
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # ----------------------------------------------------
            # Write-up section
            # ----------------------------------------------------
            with st.container(key=f"writeup_{i}"):

                st.html(
                    """
                    <style>
                        [class*="st-key-writeup_"]
                        [data-testid="stMarkdownContainer"] p,
                        [class*="st-key-writeup_"]
                        [data-testid="stMarkdownContainer"] li {
                            font-size: 1.08rem !important;
                        }
                    </style>
                    """
                )

                # ------------------------------------------------
                # Class + severity
                # ------------------------------------------------
                sev = (
                    (diag.get("severity") or "").lower()
                    if diag
                    else ""
                )

                sev_color = SEVERITY_COLOR.get(
                    sev,
                    "#57606a"
                )

                sev_emoji = SEVERITY_EMOJI.get(
                    sev,
                    "⚪"
                )

                cls_html = f"🔧 {cls}"

                if cls_conf is not None:
                    cls_html += (
                        f"&nbsp;&nbsp;"
                        f"<span style="
                        f"'color:{confidence_color(cls_conf)};"
                        f"font-weight:600;'>"
                        f"({cls_conf:.2f})"
                        f"</span>"
                    )

                st.markdown(
                    f"""
                    <div class="result-header">
                        {cls_html}
                    </div>

                    <div class="result-header"
                        style="
                            color:{sev_color};
                            margin-top:0.4rem;
                        ">
                        {sev_emoji} Severity: {sev or 'n/a'}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ====================================================
            # LLM DIAGNOSTIC OUTPUT
            # ====================================================

            if diag:

                badge = (
                    "🟢 live"
                    if source == "live"
                    else "🟡 cached"
                )

                st.caption(
                    f"source: {badge}"
                )

                st.markdown("**Diagnostic note**")

                b1, b2, b3 = st.columns(
                    3,
                    gap="small"
                )

                with b1:
                    st.html(
                        f"""
                        <div class="diagnostic-card">
                            <div class="diagnostic-card-title">
                                Likely cause
                            </div>

                            <div class="diagnostic-card-text">
                                {diag.get(
                                    "likely_cause",
                                    "—"
                                )}
                            </div>
                        </div>
                        """
                    )

                with b2:
                    st.html(
                        f"""
                        <div class="diagnostic-card">
                            <div class="diagnostic-card-title">
                                Recommended action
                            </div>

                            <div class="diagnostic-card-text">
                                {diag.get(
                                    "recommended_action",
                                    "—"
                                )}
                            </div>
                        </div>
                        """
                    )

                with b3:
                    st.html(
                        f"""
                        <div class="diagnostic-card">
                            <div class="diagnostic-card-title">
                                Summary
                            </div>

                            <div class="diagnostic-card-text">
                                {diag.get(
                                    "summary",
                                    "—"
                                )}
                            </div>
                        """
                    )

            else:
                st.warning(
                    "No diagnosis available "
                    "(no live key and no cached example)."
                )


            # ====================================================
            # DETERMINISTIC STRUCTURED ATTRIBUTES (T3)
            # ====================================================

            with st.container(key=f"t3_block_{i}"):

                with st.expander(
                    "📋  Deterministic structured attributes (T3)",
                    expanded=False
                ):

                    st.markdown(
                        """
                        <div class="t3-content">
                            <div class="attributes-heading">
                                Extracted attributes
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.json(
                        {
                            k: v
                            for k, v in attrs.items()
                            if k != "_raw"
                        }
                    )


# ===== TAB 2: analytics =====
elif page == "📊 Analytics":
    st.subheader("Segmentation: supervised (ours) vs paper zero-shot")
    summ = load_seg_summary()
    if summ:
        import pandas as pd
        paper = {"IoU": 37.49, "F1max": 53.36, "AUROC": 88.21}
        rows = []
        for metric in ["IoU", "F1max", "AUROC"]:
            rows.append({
                "Metric": metric,
                "Paper (zero-shot)": paper[metric],
                "U-Net (ours)": f"{summ['unet'][metric][0]} ± {summ['unet'][metric][1]}",
                "DeepLabV3+ (ours)": f"{summ['deeplabv3plus'][metric][0]} ± {summ['deeplabv3plus'][metric][1]}",
            })
        st.table(pd.DataFrame(rows))
        st.success(f"U-Net IoU {summ['unet']['IoU'][0]} vs paper 37.49 "
                   f"= +{summ['unet']['IoU'][0]-37.49:.1f} pts (first supervised baseline).")

    st.subheader("Per-class IoU (U-Net)")
    pc = load_per_class()
    if pc:
        import pandas as pd
        df = pd.DataFrame([{"Class": k, "IoU": round(v["IoU"]*100, 1), "n": v["n"]}
                           for k, v in pc.items()]).sort_values("IoU")
        st.bar_chart(df.set_index("Class")["IoU"])
        with st.expander("per-class table"):
            st.dataframe(df, use_container_width=True)

    st.subheader("Calibration")
    cal = load_calibration()
    if cal:
        st.metric("Expected Calibration Error (ECE)", f"{cal['ECE']}%")
        bins = [b for b in cal["bins"] if b.get("weight")]
        if bins:
            import matplotlib.pyplot as plt
            conf = [b["conf"] for b in bins]
            acc = [b["acc"] for b in bins]
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
            ax.plot(conf, acc, "o-", color="#1f77b4", label=f"model (ECE={cal['ECE']}%)")
            ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(); ax.set_aspect("equal")
            st.pyplot(fig)
            st.caption("Reliability diagram — closer to the dashed diagonal = better calibrated.")

# ===== TAB 3: eval results =====
elif page == "🧪 Eval results":
    st.subheader("LLM diagnostic evaluation")
    metrics, diagnoses = load_llm_eval()
    if metrics:
        o = metrics.get("overall", metrics)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Samples", o.get("n", len(diagnoses)))
        m2.metric("Structural validity", f"{o.get('structural_validity_pct','-')}%")
        m3.metric("Grounding rate", f"{o.get('grounding_rate_pct','-')}%")
        m4.metric("Faithfulness", f"{o.get('faithfulness_pct','-')}%")
    if diagnoses:
        st.markdown("**Browse example diagnoses:**")
        classes = sorted({d["class_name"] for d in diagnoses})
        pick = st.selectbox("Filter by class", ["(all)"] + classes)
        shown = [d for d in diagnoses if pick == "(all)" or d["class_name"] == pick]
        @st.cache_data
        def fetch_val_img(name):
            try:
                from huggingface_hub import hf_hub_download
                import cv2
                p = hf_hub_download("Zhaosxian/SteelDefectX", f"val/{name}", repo_type="dataset")
                return cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            except Exception:
                return None

        for r in shown[:30]:
            d = r["diag"]
            with st.expander(f"{r['image_name']} · {r['class_name']} · conf={r.get('confidence')}"):
                ic1, ic2 = st.columns([1, 2])
                img = fetch_val_img(r["image_name"])
                if img is not None:
                    ic1.image(img, caption=r["image_name"], clamp=True, use_container_width=True)
                ic2.markdown("**Attributes:**")
                ic2.json({
                    k: r["attributes"][k]
                    for k in ("Shape", "Scale", "Polarity", "Saliency")
                    if k in r.get("attributes", {})
                })

                ic2.markdown(f"**Cause:** {d.get('likely_cause')}")
                ic2.markdown(f"**Severity:** {d.get('severity')}")
                ic2.markdown(f"**Action:** {d.get('recommended_action')}")
                ic2.markdown(f"**Summary:** {d.get('summary')}")
    else:
        st.info("Run the LLM diagnostic notebook to populate results/llm/.")