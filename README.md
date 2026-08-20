# SteelDefectX — Explainable Steel Surface Defect Detection

An end-to-end, deployable system for steel surface defect inspection that goes beyond detection to
**explanation**: it segments defects, quantifies how confident it is, and generates a grounded
**diagnostic note** (likely cause + recommended action) for each defect — surfaced through an
interactive dashboard.

Built as a Bachelor's Thesis Project on top of the **SteelDefectX** benchmark
(Zhao, Gui, Yu & Tao, 2026), this work extends the benchmark from *description* to *deployment*:
the first supervised segmentation baseline on the dataset, calibrated uncertainty, and an LLM
diagnostic layer that the original paper flagged as future work.

---

## Headline results

| Component | Metric | This work | Reference |
|---|---|---|---|
| **Segmentation** (U-Net) | IoU | **78.51 ± 0.30** | 37.49 (paper, zero-shot) |
| **Segmentation** (DeepLabV3+) | IoU | 75.22 ± 0.23 | 37.49 (paper, zero-shot) |
| **Classification** (ResNet34) | Accuracy | **97.4%** | ~94% (paper, CLIP-Adapter) |
| **Uncertainty** (dropout U-Net) | ECE | **1.31%** | — |
| **LLM diagnostics** (250 samples) | Validity / Grounding / Faithfulness | **100% / 100% / 100%** | — |

> **Framing note (important):** the paper only ever evaluated segmentation *zero-shot* (best 37.49 IoU
> via a vision–language model); it never trained a model on the pixel masks. This project establishes
> the **first supervised segmentation baseline**, so the comparison is "supervised baseline vs. the
> paper's zero-shot ceiling," not "our model beats their model." Likewise, the classifier is *comparable
> to* the paper's CLIP-Adapter result under a different setup — not a strict head-to-head.

---

## What the system does

```
 Steel image
     │
     ├─► Segmentation (U-Net, ResNet34 encoder)  ──►  defect mask
     │
     ├─► MC-Dropout (30 stochastic passes)  ──►  per-pixel uncertainty + confidence
     │
     ├─► Classifier (ResNet34)  ──►  auto-assigned defect class (25 classes)
     │
     ├─► Deterministic attribute extraction  ──►  shape / scale / polarity / saliency / position …
     │
     └─► LLM diagnostic layer (grounded in the dataset's cause vocabulary)
                    │
                    └─►  { likely_cause, severity, recommended_action, summary }
                         + human-review flag when confidence is low
```

Everything is unified in a **Streamlit dashboard**: upload an image → see the mask, uncertainty
heatmap, auto-assigned class, structured attributes, and a diagnostic note, with an aggregate
analytics view of all results.

---

## Key ideas

- **First supervised segmentation baseline.** An ImageNet-pretrained ResNet34 encoder (fully
  fine-tuned) with U-Net / DeepLabV3+ decoders, trained on the pixel masks with a combined **Dice + BCE**
  loss. Both architectures share the same encoder, so the head-to-head isolates the decoder. U-Net wins
  (78.5 vs 75.2 IoU) — its skip connections preserve the fine detail that thin/small defects need.

- **Calibrated uncertainty for deployment trust.** A dropout-enabled U-Net variant runs **MC-Dropout**
  at inference to produce per-pixel uncertainty. Uncertainty concentrates on defect *boundaries* and
  faint/ambiguous regions (exactly where a human would hesitate), and the model is well-calibrated
  (**ECE 1.31%**). Low-confidence predictions are flagged for human review.

- **Diagnostic, not just descriptive, explanation.** The LLM layer never sees the raw image — it reasons
  over *deterministically computed* attributes plus the defect class's real industrial cause, and is
  constrained to produce a grounded cause → action note. Evaluated on 250 stratified samples
  (10 per class × 25 classes): **100% structural validity, 100% cause-grounding, 100% faithfulness** —
  quantitative, hallucination-free diagnostic output.

- **Reproducibility built in.** Frozen stratified data split (seed 42), 3-seed segmentation runs with
  mean ± std, unit-tested metrics, and resumable training harnesses.

---

## Results in detail

### Segmentation (3 seeds each, held-out val set)

| Model | IoU | F1-max | AUROC |
|---|---|---|---|
| **U-Net (ResNet34)** | **78.51 ± 0.30** | 87.97 ± 0.19 | 97.70 ± 0.09 |
| DeepLabV3+ (ResNet34) | 75.22 ± 0.23 | 85.90 ± 0.14 | 97.29 ± 0.14 |
| Paper (zero-shot best) | 37.49 | 53.36 | 88.21 |

Per-class IoU reveals the difficulty lives in the **data**, not the model: both architectures excel on
sharp, well-defined defects (Punching ~0.95, Crescent gap ~0.94) and both struggle on diffuse/rare ones
(Waist folding ~0.44, Rolled pit ~0.47, Crazing ~0.51).

### Classification (ResNet34, 2,324 val images, 25 classes)

Accuracy **97.4%**, macro-F1 **96.9%** — comparable to the paper's ~94% CLIP-Adapter result under a
different setup. Used to auto-assign the class so the diagnostic pipeline runs end-to-end.

### Uncertainty & calibration

ECE **1.31%** (well-calibrated), with mild mid-range overconfidence typical of Dice+BCE segmentation.
See `results/uncertainty/uncertainty_examples.png` for MC-Dropout heatmaps.

### LLM diagnostics

250 diagnoses (10 per class), **100%** on all three automated metrics. Example (Iron scale compression):
cause → *"oxide scale mechanically embedded into the hot steel surface during rolling"*; action →
*"inspect high-pressure descaling header nozzles for clogs; check entry temperature at the rolling mill."*

---

## Repository structure

```
SteelDefectX/
├── src/
│   ├── data/            make_split.py (frozen stratified split), dataset.py
│   ├── segmentation/    model.py, losses.py (Dice+BCE), metrics.py, train.py,
│   │                    uncertainty.py (MC-Dropout, ECE), classifier.py
│   ├── llm/             attributes.py (deterministic T3), diagnose.py, evaluate_llm.py
│   └── evaluation/      aggregate.py (head-to-head + per-class tables)
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_train_segmentation.ipynb / 02c_..._kaggle.ipynb
│   ├── 03_uncertainty_calibration.ipynb
│   ├── 04_llm_diagnostic_local.ipynb
│   └── 05_train_classifier.ipynb
├── dashboard/app.py     Streamlit app (inspect + analytics)
├── results/             segmentation / uncertainty / llm / classification metrics (JSON)
└── requirements.txt
```

Model weights (`models/*.pt`) and the dataset (`data/`) are gitignored — see setup below.

---

## Setup

```bash
git clone https://github.com/d-mondal/SteelDefectX.git
cd SteelDefectX
python -m venv .venv && .venv\Scripts\activate      # Windows (use source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

> **Torch on Windows:** if you hit a `c10.dll` load error, install the CPU build explicitly
> (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) and ensure the MSVC++
> redistributable is installed.

**Dataset** (25 classes, 7,778 images with pixel masks + text annotations) — clone from Hugging Face:

```bash
git clone https://huggingface.co/datasets/Zhaosxian/SteelDefectX sdx_data
```

**API key** for the LLM layer — create a `.env` at the repo root:

```
GEMINI_API_KEY=your-key-here
```

(free tier at [aistudio.google.com](https://aistudio.google.com); `.env` is gitignored.)

---

## Running it

**Train** (Kaggle T4 recommended; notebooks are resumable):

```
notebooks/02c_train_segmentation_kaggle.ipynb   # segmentation sweep
notebooks/03_uncertainty_calibration.ipynb      # dropout U-Net + ECE
notebooks/05_train_classifier.ipynb             # classifier
notebooks/04_llm_diagnostic_local.ipynb         # LLM diagnostics + eval
```

**Dashboard** (needs `models/unet_dropout.pt`, `models/classifier.pt`, and `results/`):

```bash
streamlit run dashboard/app.py
```

---

## Tech stack

PyTorch · segmentation-models-pytorch · MC-Dropout · scikit-learn · Google Gemini API (`google-genai`)
· Streamlit · Hugging Face Hub

## Acknowledgements

Built on the **SteelDefectX** dataset and benchmark by Zhao, Gui, Yu & Tao (2026), arXiv:2603.21824.
This project extends it with supervised segmentation, uncertainty quantification, and a diagnostic LLM
layer; all "paper" numbers refer to results reported therein and are used for context, with the framing
caveats noted above.