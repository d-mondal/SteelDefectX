# CLAUDE.md — Project context for Claude Code

This file is read automatically at the start of every Claude Code session. It gives you (Claude Code)
persistent context so the user doesn't have to re-explain the project. Read it fully before acting.

## What this project is

A Bachelor's Thesis Project (BTP): a **supervised, explainable steel surface defect detection system**
built on the **SteelDefectX** dataset (Zhao, Gui, Yu & Tao, 2026 — arXiv 2603.21824). It is NOT a
reproduction of the paper. It extends the paper into a deployable system.

The paper published a dataset + annotation pipeline only — **no model training/eval code exists upstream.**
Our contribution is the modelling and system layer the paper never built.

## The core contribution (state this framing exactly; it's load-bearing for the CV and the report)

The paper only ever **zero-shot / prompt-evaluates** segmentation (best IoU 37.49, via AnomalyCLIP learned
prompts). It **never trains a model on the pixel masks.** We train the **first mask-supervised segmentation
model** on SteelDefectX.

- Always frame results as: "first supervised baseline, improving over the paper's best **zero-shot** IoU of
  37.49 by X points" — **never** "we beat their model" (not apples-to-apples: supervised vs zero-shot).
- Classification is a **sanity check only** — the paper is already at ~94% Acc (CLIP-Adapter fine-tuning,
  not zero-shot). Do not chase classification gains; that's not the story.

## Locked decisions (do not silently revisit these)

- **Masks are BINARY** (defect vs background), confirmed from the HF README. So: train ONE shared binary
  segmentation model across all 25 classes. Recover **per-class IoU/Dice at eval time** using each image's
  known class label. Do NOT attempt 25-way pixel semantic segmentation, and do NOT train per-class models
  (rare classes have as few as ~50 images).
- **Active Learning is OUT OF SCOPE** for this project. It lives in a separate property-prediction/GNN
  project. Do not add it here.
- **No RAG.** The paper's T1 class-level "industrial causes" + our own deterministic T3-style attributes are
  already a small complete knowledge base. A vector DB adds overhead without differentiation.
- **No foundation-model pretraining from scratch.** Segmentation uses an ImageNet-pretrained encoder +
  trained decoder. That's fine-tuning, not pretraining.
- Retain **all 25 classes including rare ones** (crease, rolled pit, etc.) in evaluation.

## Data (VERIFIED via Day-0 audit, not inferred)

- Lives on Hugging Face: `huggingface.co/datasets/Zhaosxian/SteelDefectX` (NOT in git — see .gitignore).
- Layout: `train/ train_mask/ train-text.json  val/ val_mask/ val-text.json  class_descriptions.json`
- **Counts confirmed from metadata:** 5,454 train + 2,324 val = **7,778** total (matches paper). Split = 70/30.
- There is **no separate test set** upstream. Plan: treat their val split as our test set; carve a small
  internal val set out of train for our own model selection. Document this.
- 25 classes, 256×256, long-tailed.
- Masks: binary PNG, foreground = defect, threshold at `pixel > 127`.
- text JSONs are **lists** of per-image entries (train-text.json = 5454, val-text.json = 2324); each entry
  carries image_name, class_name, natural_language_description (T2), structured_attributes (T3), template_sentence (T4).

## The 25 classes + their T1 industrial causes (VERIFIED — this is the diagnostic layer's grounding vocabulary)

class_descriptions.json ships ONE sentence per class of the form "A photo of <class>, which shows
<visual attributes> caused by <cause>." The causal clause is what the diagnostic LLM layer selects/reasons over.
NOTE: T1 is thinner than the paper implied (single fused sentence, not a rich structured cause taxonomy) —
state it accurately in the report; don't overclaim the knowledge base.

- Bright scratch — friction with foreign objects
- Crazing — uneven cooling or residual stress
- Crease — localized yielding during uncoiling
- Crescent gap — cutting during strip production
- Dark scratches — mechanical abrasion or roller damage
- Finishing roll printing — slippage between work rolls and strip during finishing
- Inclusion — smelting residue embedded during rolling
- Iron scale compression — dark gray iron scale embedded into surface
- Iron sheet ash — accumulated metallic dust and oil
- Oil spot — lubricant contamination
- Oxide scale of plate system — embedded oxide particles from roller damage during high-speed hot rolling
- Oxide scale of temperature system — excessive temperature or uneven cooling
- Patches — uneven processing
- Pitted surface — localized corrosion
- Punching — unintended punching from equipment malfunction
- Red iron sheet — high-silicon steel oxidation during heating
- Rolled in scale — scale material rolled into base metal
- Rolled pit — damaged work rolls or tension roller defects
- Secondary rust skin — prolonged exposure (flaky corrosion)
- Silk spot — uneven roll pressure or temperature
- Slag inclusion — exposed slag during hot rolling
- Waist folding — severe local deformation due to low carbon content
- Water spot — uneven drying
- Welding line — coil welding during strip transition
- White rust — zinc corrosion under humid conditions

**Close/confusable pairs to watch in per-class reporting & diagnostic disambiguation** (lean on distinct causes):
Oxide scale of plate system ↔ Oxide scale of temperature system; Rolled in scale ↔ Iron scale compression ↔ Slag inclusion.

## Reuse the upstream deterministic attribute code — don't reinvent it

The original repo's `data_analysis.py` already computes scale/polarity/saliency/component-count/position from
image+mask AND checks image↔mask pairing integrity. Reuse it for the Day-0 gate and as the basis of our
diagnostic layer. Exact T3 discretization thresholds (from upstream `build_structured_text_annotations.py`):

```
scale_ratio → Scale:  <0.01 tiny | <0.03 small | <0.10 medium | <=0.25 large | else extensive
polarity_diff → Polarity: <-5 dark | >5 bright | else neutral   (= mean_defect − mean_background)
saliency → Saliency:  <10 low | <=40 medium | else high         (= |polarity_diff|)
component_count → Number: 1..5 one..five | else "multiple"       (8-connectivity)
Position: 3×3 grid, per-cell defect-pixel fraction, ranked, cells with >0 kept
```

## The pipeline we are building

```
data → preprocessing (256², 7:3 split, internal val carve-out)
     → [sanity check] 1–2 CLIP-Adapter classification configs (cite Table 2)
     → OUR supervised binary segmentation (U-Net / DeepLabV3+, Dice+BCE, 3 seeds)  ← FLAGSHIP
     → uncertainty (MC-Dropout) + calibration (ECE, reliability diagram)
     → deterministic T3-style attribute extraction from OUR predicted masks
     → LLM diagnostic layer (constrained to T1 cause vocabulary; diagnostic, not descriptive)
     → Streamlit dashboard
```

## Guardrails for you (Claude Code)

- **Scoping calls belong to the user + the planning chat, not you.** Which classes, how to frame CV claims,
  whether to add a component — surface options, don't decide unilaterally.
- Prefer reusing upstream code (`data_analysis.py`) over rewriting.
- Every new metric result on our supervised model must be **3-seed, mean ± std** (the paper's low-variance
  claim was classification-only; segmentation is a genuinely new result and needs its own variance).
- Keep data/, models/, outputs/ out of git (already in .gitignore). Never commit large binaries.
- Use Weights & Biases for experiment tracking and Streamlit for the dashboard (user's existing stack).

## Repo structure to grow into

```
src/{preprocessing,paper_baseline,segmentation,uncertainty,attributes,llm,evaluation}/
dashboard/  configs/  models/  results/  reports/  notebooks/
```

## Current status

- [x] Repos reviewed, scope locked, go/no-go = conditional GO
- [x] Local env rebuilt on Python 3.12.10 (matches Colab); core tooling installed
- [x] **Day-0 go/no-go gate = PASSED.** Evidence: metadata confirms 7778 (5454 train + 2324 val), 25 classes
      with real names + T1 causes; T3 schema verified (Position = ranked [cell, fraction] list); sample of 40
      pairs had 0 missing masks (naming = train_mask/<stem>.png); masks binary (27/40 strictly, 13 anti-aliased
      edges that binarize cleanly at >127); overlays aligned incl. rarest class Rolled pit (35 imgs).
      Rarest 3 classes: Rolled pit (35), Oxide scale of plate system (45), Crease (53).
- [ ] **NEXT: frozen split** — carve internal val from train (stratified by class), keep official val as test
- [ ] Segmentation baseline (U-Net / DeepLabV3+, Dice+BCE, 3 seeds) ← FLAGSHIP, starts now
- [ ] Segmentation baseline (hard checkpoint)
- [ ] Uncertainty + calibration
- [ ] Attribute extraction + LLM diagnostic layer
- [ ] Dashboard
