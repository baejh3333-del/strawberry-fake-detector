<div align="center">

<img src="assets/social-preview.png" width="100%" alt="Strawberry Fake Detector: real or AI-generated?">

[Overview](#overview) · [Results](#results) · [Architecture](#architecture) · [Data pipeline](#data-pipeline) · [What went wrong](docs/lessons-learned.md) · [한국어](README.ko.md)

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c) ![Backbone](https://img.shields.io/badge/backbone-DINOv3%20ViT--L%2F16-green) ![Accuracy](https://img.shields.io/badge/leaderboard%20accuracy-0.87-brightgreen) ![License](https://img.shields.io/badge/code-MIT-lightgrey)

</div>

## Overview

Generative AI has made crop photos easy to fabricate or retouch, which weakens the trust we can place in image-based agricultural records. This project tackles a narrow, testable version of that problem, as posed by the 1st National Agricultural AI Competition (Korea, image mission): *is this greenhouse strawberry photo a **real photograph (0)** or **generated / composited / edited (1)**?*

**Why it is hard**

- **No training data.** Only 21 example images were released, so the whole training set had to be built from scratch.
- **Six kinds of fakes**, from fully generated images to local edits that change only a few percent of a photo.
- **Unknown generators.** The test fakes come from methods the model has never seen, so scores on my own validation set overestimated the hidden test (0.874 vs. 0.77 at first).
- **Hard runtime limits.** Offline inference, 5 minutes for 524 images on a T4, ≤ 1 GB.

**What this repository contributes**

1. A reproducible pipeline that turns real photos into every kind of fake with 15+ generators and filters, giving a training pool of about 17k images ([`data_pipeline/`](data_pipeline)).
2. A detector that keeps pixel-level traces (native-resolution tiles and a frozen DINOv3) and models generator-specific fingerprints (Experts + Gate) next to a full-image CNN.
3. An evaluation protocol (leave-one-generator-out, plant-level splits) that explains the gap between own validation and the hidden test.
4. A candid log of what did *not* work and why ([docs/lessons-learned.md](docs/lessons-learned.md)).

## At a glance

| | |
|---|---|
| **Project** | Two deep-learning beginners, four weeks (2026-09-21 to 10-16) |
| **Final score** | **0.87 accuracy** on the hidden 524-image test (262 real / 262 fake) |
| **Model** | 311M parameters, 0.64 GB, **152 s** for 524 images on a Colab T4 (limit: 5 min, offline, ≤ 1 GB) |
| **Key idea** | Tiles at native resolution → frozen DINOv3 → attention-pooled **Experts per generator** + a full-image CNN that also sees a high-pass residual |
| **Most useful finding** | Own validation said 0.874, the hidden test said 0.77. Leave-one-generator-out evaluation explained why. |

## Results

<img src="assets/results.png" width="62%" alt="Validation vs leaderboard accuracy">

| Submission | Change | Own validation | Leaderboard |
|---|---|---|---|
| exp02 | first full training | acc 0.874 / AUROC 0.958 | 0.77 |
| exp06 | re-assign "clone / local filter" labels | acc 0.875 | ≈ 0.75 |
| exp11b (v3) | one Expert per generation method (24), drop 11 inert methods | acc 0.885 / AUROC 0.964 | 0.83 |
| v5 | exp11b weights, decision threshold p ≥ 0.05 (`+ln 19` on the last bias) | acc 0.901 | 0.85 |
| **v7 (final)** | exp11b setup + **high-pass residual input** to the Common CNN (exp16), same threshold shift | acc 0.902 | **0.87** |

On my own validation, none of seven architecture ideas (LayerNorm fusion, top-k pooling, high-pass input, dropout, FFT features, …) moved the metric beyond noise. One of them, the **high-pass residual input** (exp16), ended up in the best leaderboard model: v7 scored 0.87 vs. 0.85 for v5, although validation rated the two the same (balanced accuracy 0.814 vs. 0.813). That gap is about 10 of 524 images (standard error ≈ 1.5 pp), so one comparison neither proves nor refutes the idea. What stayed consistent was the bottleneck: **how many generators the model had seen**, not model capacity.

<img src="assets/experiments.png" width="62%" alt="Experiment deltas">

> One run per experiment (no seed repeats). The shaded band is a rough noise level of ±0.005 seen between comparable runs, not a measured confidence interval. Amber = the idea that ended up in the final model. Full table: [docs/experiments.md](docs/experiments.md).

## Architecture

<img src="assets/architecture.png" width="92%" alt="Detector architecture">

| Design choice | Why |
|---|---|
| **12 native-resolution tiles, no resizing** | Blur, VAE 8-px grids and edit boundaries are pixel-level traces that downscaling destroys. |
| **Frozen DINOv3** | A few thousand images per generator: fine-tuning would memorize *my* generators' fingerprints. |
| **Attention pooling** | A local edit lives in a few tiles; mean pooling washes it out. |
| **Common CNN on the full image** | Covers high-frequency traces (noise, JPEG) and has no input-size limit. The final model feeds it RGB plus a high-pass residual (6 channels). |
| **One Expert per generation method + Gate** | Different generators leave different fingerprints. |

Training: (1) Experts + Common CNN → (2) Gate → (3) everything, with `BCE + 0.2·GateCE + 0.1·SupCon` (SupCon pairs a real photo with fakes made from it). Details: [docs/architecture.md](docs/architecture.md).

## Data pipeline

<img src="assets/pipeline.png" width="92%" alt="Data pipeline">

About 17k training images from 15+ generators. Train/val are split by **plant ID** to avoid leakage, and "leave-one-generator-out" folds measure generalization to unseen methods. Sources, licenses and prompts: [docs/data-and-generation.md](docs/data-and-generation.md).

## Quick start

```bash
pip install -r requirements.txt

# 1) Smoke-test the inference package with random weights (no data needed)
python tools/smoke_test_submission.py

# 2) Train (needs a manifest + images produced by data_pipeline/)
python train/train.py --data <DATA_DIR> --pretrained <PRETRAINED_DIR> --out <RUN_DIR> --stage all --family_map gen24
```

> **Note:** `submission/model/config.json` is the default template (5 Experts). Training with `--family_map gen24` writes `n_experts: 24` into the exported config, which is what the final model used.
>
> **Not included:** training images, the contest data and trained weights. Real photos come from [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71451) (personal-use terms, no redistribution); fakes are regenerated with `data_pipeline/`. The DINOv3 backbone is downloaded separately under the [DINOv3 License](submission/model/LICENSE_DINOv3.md).

## Repository layout

```
submission/       inference package as submitted (detector.py, inference.py, model/config.json)
train/            3-stage trainer, unit tests, diagnostics (LOGO, threshold scan, expert usage)
data_pipeline/    01–16: real extraction → filters → composites → VAE recon → inpaint → t2i → style → local edits
tools/            JPEG scanner, image normalizer, submission smoke test
assets/           figures (regenerate with assets/make_figures.py)
docs/             architecture · data-and-generation · experiments · lessons-learned  (written in Korean)
```

## Known limitations

Fakes from an **unseen AI editing model** (partial-generation recall 0.32 on unseen methods), FLUX-VAE reconstructions (recall ≈ 0), high-quality JPEG-recompression fakes, and real photos of varieties absent from training (shortcut: "AI Hub style ⇒ real"). Analysis in [docs/lessons-learned.md](docs/lessons-learned.md).

## Team

[@baejh3333-del](https://github.com/baejh3333-del) · [@pjo789412-afk](https://github.com/pjo789412-afk)

## Acknowledgements

[DINOv3](https://github.com/facebookresearch/dinov3) (Meta), [timm](https://github.com/huggingface/pytorch-image-models), [Hugging Face Transformers / Diffusers](https://github.com/huggingface), [LaMa](https://github.com/advimman/lama), FLUX, Stable Diffusion XL, Z-Image, Infinity, Kandinsky, and the AI Hub strawberry dataset.

## Citation

```bibtex
@misc{strawberry_fake_detector_2026,
  title  = {Strawberry Fake Detector: DINOv3 + Mixture-of-Experts for real vs. AI-generated crop images},
  author = {baejh3333-del and pjo789412-afk},
  year   = {2026},
  url    = {https://github.com/baejh3333-del/strawberry-fake-detector}
}
```

## License

Code: [MIT](LICENSE). Third-party models and datasets keep their own licenses ([details](docs/data-and-generation.md)).
