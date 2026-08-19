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

# make 'src' importable when run from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="SteelDefectX Inspector", layout="wide")

CKPT = str(ROOT / "models" / "unet_dropout.pt")
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


def predict(model, device, gray, mc=20):
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


# ------------------------- UI -------------------------
st.title("🔩 SteelDefectX — Explainable Defect Inspector")
st.caption("Supervised segmentation · calibrated uncertainty · LLM diagnostic reasoning")

tab1, tab2, tab3 = st.tabs(["🔍 Inspect", "📊 Analytics", "🧪 Eval results"])

# ===== TAB 1: single-image pipeline =====
with tab1:
    import cv2
    st.subheader("Upload a steel surface image")
    up = st.file_uploader("PNG / JPG", type=["png", "jpg", "jpeg", "bmp"])
    known_class = st.selectbox(
        "Defect class (for the diagnostic cause lookup)",
        options=["(auto: unknown)"] + sorted(load_t1().keys()),
    )
    go = st.button("Run inspection", type="primary", disabled=up is None)

    if go and up is not None:
        file_bytes = np.frombuffer(up.read(), np.uint8)
        gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, (256, 256))

        with st.spinner("Running segmentation + MC-Dropout uncertainty..."):
            model, device = load_model()
            prob, unc, conf = predict(model, device, gray)

        c1, c2, c3 = st.columns(3)
        c1.image(gray, caption="Input", clamp=True, use_container_width=True)
        c2.image((prob > 0.5).astype(float), caption="Predicted mask", clamp=True, use_container_width=True)
        # uncertainty heatmap (normalize for display)
        un = (unc - unc.min()) / (np.ptp(unc) + 1e-8)
        c3.image(un, caption="Uncertainty (MC-Dropout std)", clamp=True, use_container_width=True)

        # confidence + human-review flag
        low_conf = conf < 0.6
        st.metric("Model confidence", f"{conf:.2f}",
                  delta="⚠️ LOW — human review advised" if low_conf else "OK",
                  delta_color="inverse" if low_conf else "normal")

        # deterministic T3 attributes
        from src.llm.attributes import extract_attributes
        attrs = extract_attributes(gray, prob)
        st.markdown("**Deterministic structured attributes (T3-style):**")
        st.json({k: v for k, v in attrs.items() if k != "_raw"})

        # LLM diagnosis (live -> cached fallback)
        t1 = load_t1()
        cls = None if known_class.startswith("(auto") else known_class
        _, diagnoses = load_llm_eval()
        st.markdown("**Diagnostic note:**")
        if cls is None:
            st.info("Select the defect class above to ground the diagnostic cause in the dataset's "
                    "industrial-cause vocabulary. (Class prediction could be added via the classifier.)")
        else:
            diag, source = None, None
            try:
                diag = try_live_llm(cls, t1[cls], attrs, conf, float(unc.mean()))
                source = "live"
            except Exception as e:
                diag = cached_llm_for_class(cls, diagnoses)
                source = "cached (live unavailable)"
            if diag:
                badge = "🟢 live" if source == "live" else "🟡 cached"
                st.caption(f"source: {badge}")
                cc1, cc2 = st.columns([2, 1])
                cc1.markdown(f"**Likely cause:** {diag.get('likely_cause')}")
                cc1.markdown(f"**Recommended action:** {diag.get('recommended_action')}")
                cc1.markdown(f"**Summary:** {diag.get('summary')}")
                sev = (diag.get("severity") or "").lower()
                color = {"high": "🔴", "moderate": "🟠", "low": "🟢"}.get(sev, "⚪")
                cc2.markdown(f"**Severity**\n\n# {color} {sev or 'n/a'}")
            else:
                st.warning("No diagnosis available (no live key and no cached example).")

# ===== TAB 2: analytics =====
with tab2:
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
            import pandas as pd
            rel = pd.DataFrame({"confidence": [b["conf"] for b in bins],
                                "accuracy": [b["acc"] for b in bins]}).set_index("confidence")
            st.line_chart(rel)
            st.caption("Reliability diagram — closer to the diagonal = better calibrated.")

# ===== TAB 3: eval results =====
with tab3:
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
        for r in shown[:30]:
            d = r["diag"]
            with st.expander(f"{r['image_name']} · {r['class_name']} · conf={r.get('confidence')}"):
                st.write("**Attributes:**", {k: r["attributes"][k] for k in
                         ("Shape", "Scale", "Polarity", "Saliency") if k in r.get("attributes", {})})
                st.write("**Cause:**", d.get("likely_cause"))
                st.write("**Severity:**", d.get("severity"))
                st.write("**Action:**", d.get("recommended_action"))
                st.write("**Summary:**", d.get("summary"))
    else:
        st.info("Run the LLM diagnostic notebook to populate results/llm/.")
