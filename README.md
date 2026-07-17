# Robust Early Intent Prediction for Human-Robot Collaboration
## A Multi-Model Consensus Approach under Partial Occlusion

**Authors:** Soumitra Chowdhury, Mohammad Tausiful Alam

**Supervisor:** Antu Chowdhury, Lecturer, SoSET

**University:** East Delta University, Bangladesh


---

## Overview

In industrial Human-Robot Collaboration (HRC), robots must predict human intent **before the action is completed** to act safely and efficiently. A critical real-world challenge is **partial occlusion** — boxes, equipment, or the robot's own arm blocking the camera's view of the worker — which causes skeleton joints to go missing and makes single-model predictions unreliable.

This work proposes a **consensus-based inference framework** that runs two skeleton-based action recognition models in parallel and fuses their predictions using entropy-weighted confidence scoring. When one model is confused by missing joints, the other can compensate — and the fusion layer knows which model to trust at each moment.

> **Note:** The methodology diagram will be added here once finalised in draw.io.

---

## Key Results

### HRI30 — Occlusion Robustness Benchmark (Primary Contribution)

| Occlusion | ST-GCN | CTR-GCN | **Consensus (Ours)** | Improvement |
|-----------|--------|---------|----------------------|-------------|
| 0%        | 61.22% | 69.05%  | **70.75%**           | +1.70 pp    |
| 10%       | 49.05% | 52.76%  | **57.24%**           | +4.48 pp    |
| 20%       | 40.48% | 37.04%  | **46.29%**           | +5.81 pp    |
| 30%       | 39.01% | 26.94%  | **40.17%**           | +1.16 pp    |
| 50%       | 23.47% | 16.22%  | **24.15%**           | +0.68 pp    |

Accuracy = Top-1. Improvement shown over the stronger individual model at each occlusion rate.

### Extended Benchmark — NTU RGB+D Verification

| Dataset              | Model   | Top-1 (Ours) | Top-1 (Paper) |
|----------------------|---------|--------------|---------------|
| NTU RGB+D 60 X-Sub   | ST-GCN  | 81.28%       | 81.5%         |
| NTU RGB+D 60 X-Sub   | CTR-GCN | 89.94%       | 89.9%         |
| NTU RGB+D 120 X-Sub  | CTR-GCN | 84.91%       | 84.9%         |

---

## System Architecture

```
                    SKELETON INPUT
         (25 NTU-style joints per frame)
                         |
                         v
             OCCLUSION SIMULATION
       (randomly zero N% of joints at inference)
                         |
            +------------+------------+
            |                         |
            v                         v
       +---------+               +----------+
       |  ST-GCN |               | CTR-GCN  |
       |  (GCN   |               | (Channel |
       | Branch) |               |  -wise)  |
       |softmax  |               | softmax  |
       |   P1    |               |   P2     |
       +---------+               +----------+
            |                         |
            +------------+------------+
                         |
                         v
           ENTROPY-WEIGHTED CONSENSUS
           H  = -sum( p * log(p) )  per model
           w  = exp(-H),  normalized per sample
           P_final = w1*P1 + w2*P2
                         |
                         v
        Predicted Intent Class + Confidence Score
        "SAFE TO PROCEED" / "WAIT - LOW CONFIDENCE"
```

**Why two models?** ST-GCN (Yan et al., AAAI 2018) uses a fixed graph topology and excels at structural motion patterns. CTR-GCN (Chen et al., ICCV 2021) dynamically refines topology per channel and handles global motion variation better. Their failure modes under occlusion are different and complementary — the consensus layer exploits this by trusting whichever model is more confident on each sample.

---

## Repository Structure

```
HRC-Intent-Prediction/
|
+-- README.md
|
+-- notebooks/
|   +-- phase2_hri30_skeleton_extraction.ipynb
|   +-- phase2_hri30_formatter.ipynb
|   +-- phase2_stgcn_finetune.ipynb
|   +-- phase2_ctrgcn_finetune.ipynb
|   +-- phase2_model_comparison.ipynb
|   +-- phase3_occlusion_benchmark.ipynb
|   +-- phase4_consensus_layer.ipynb
|   +-- phase4_5_ntu120_preprocessing.ipynb
|   +-- phase5_extended_benchmarking.ipynb
|
+-- consensus/
|   +-- consensus_layer.py
|
+-- configs/
|   +-- stgcn_hri30.yaml
|   +-- ctrgcn_hri30.yaml
|
+-- figures/
|   +-- methodology_diagram.png
|   +-- accuracy_degradation_curves.png
|   +-- consensus_vs_individual.png
|
+-- results/
|   +-- phase5_benchmark_results.csv
|   +-- occlusion_results.csv
|   +-- consensus_benchmark_results.csv
|
+-- requirements.txt
```

---

## Datasets

### HRI30 (Primary Dataset)
- 2,940 RGB video clips, 30 industrial HRC action classes, 98 samples per class
- Download: https://zenodo.org/records/5833411
- After downloading, place in `data/HRI30/`
- Skeleton extraction is handled in `phase2_hri30_skeleton_extraction.ipynb` using MediaPipe Pose, remapped from 33 MediaPipe landmarks to 25 NTU-style joints

### NTU RGB+D 60 and 120
- Request access from ROSE Lab: https://rose1.ntu.edu.sg/dataset/actionRecognition
- Download `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60) and `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120 extension)
- NTU120 preprocessing follows the official 3-step pipeline — see `phase4_5_ntu120_preprocessing.ipynb`
- The NTU RGB+D datasets require an approved access request from ROSE Lab. Raw data cannot be redistributed.

---

## Prerequisites

- Python 3.8+
- PyTorch 2.11.0 with CUDA 12.8 (tested on Tesla T4)
- Google Colab (all notebooks designed for Colab + Google Drive)

For GPU support, install PyTorch first with the correct CUDA index:

```bash
pip install torch==2.11.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

Then install all other dependencies:

```bash
pip install -r requirements.txt
```

Clone backbone repositories:

```bash
git clone https://github.com/yysijie/st-gcn
cd st-gcn && pip install -e torchlight && cd ..

git clone https://github.com/Uason-Chen/CTR-GCN
cd CTR-GCN && pip install -r requirements.txt && pip install -e torchlight && cd ..
```

---

## Pretrained Weights

| Model   | Dataset             | Top-1  | Source |
|---------|---------------------|--------|--------|
| ST-GCN  | NTU RGB+D 60 X-Sub  | 81.28% | Official ST-GCN repo (see Acknowledgements) |
| CTR-GCN | NTU RGB+D 60 X-Sub  | 89.94% | Official CTR-GCN repo (see Acknowledgements) |
| CTR-GCN | NTU RGB+D 120 X-Sub | 84.91% | Official CTR-GCN repo (see Acknowledgements) |
| ST-GCN  | HRI30 fine-tuned    | 61.22% | Available upon request |
| CTR-GCN | HRI30 fine-tuned    | 69.05% | Available upon request |

---

## How to Run

All notebooks are self-contained and run sequentially in Google Colab. Each notebook mounts Google Drive at the start and saves outputs there. Run in this exact order:

1. `phase2_hri30_skeleton_extraction.ipynb` — Extract skeletons from HRI30 video clips (CPU runtime, run once, ~4 hours)
2. `phase2_hri30_formatter.ipynb` — Format skeletons into ST-GCN and CTR-GCN input format (CPU)
3. `phase2_stgcn_finetune.ipynb` — Fine-tune ST-GCN on HRI30 (GPU T4, ~2 hours)
4. `phase2_ctrgcn_finetune.ipynb` — Fine-tune CTR-GCN on HRI30 (GPU T4, ~3 hours)
5. `phase2_model_comparison.ipynb` — Compare per-class accuracy of both models (CPU)
6. `phase3_occlusion_benchmark.ipynb` — Run occlusion robustness benchmark (GPU T4, ~30 min)
7. `phase4_consensus_layer.ipynb` — Run consensus fusion benchmark (GPU T4, ~45 min)
8. `phase4_5_ntu120_preprocessing.ipynb` — Preprocess NTU RGB+D 120 (CPU, ~2-3 hours)
9. `phase5_extended_benchmarking.ipynb` — Verify results on NTU60 and NTU120 (GPU T4, ~30 min)

### Using the consensus layer standalone

```python
from consensus.consensus_layer import consensus_fusion, apply_occlusion

data_occluded = apply_occlusion(data_np, occlusion_rate=0.20, seed=42)

logits_st  = get_model_logits(stgcn_model, data_occluded, device)
logits_ctr = get_model_logits(ctrgcn_model, data_occluded, device)

predictions = consensus_fusion(logits_st, logits_ctr)
```

---

## Important Notes

1. **No geometric augmentation on HRI30.** HRI30 contains directional action classes such as handover direction variants. Rotation or flip augmentation was empirically tested and reduced Top-1 accuracy from 61.22% to 55.10%.

2. **Consensus formula.** Weight per model is `w = exp(-H)` where H is Shannon entropy of the softmax output, normalized so weights sum to 1 per sample. Do not substitute `w = 1/(H+epsilon)` — this is a different formula with different behaviour.

3. **NTU120 preprocessing must follow the official pipeline.** Skipping skeleton centering, actor duplication for single-person clips, or using the wrong subject split produces approximately 11% accuracy instead of 85%. See `phase4_5_ntu120_preprocessing.ipynb`.

4. **ST-GCN requires a torch.load patch.** PyTorch 2.x rejects 2018-era checkpoints by default. All ST-GCN notebooks apply this patch automatically at the start.

5. **CTR-GCN config paths must be set in YAML directly.** Do not pass `data_path` via `--test-feeder-args` on the command line — the CLI parser breaks on file paths containing forward slashes.

---

## Acknowledgements

We thank our supervisor **Antu Chowdhury** (Lecturer, SoSET, East Delta University) for guidance and support throughout this research project.

This work builds on:
- ST-GCN: https://github.com/yysijie/st-gcn (Yan et al., AAAI 2018)
- CTR-GCN: https://github.com/Uason-Chen/CTR-GCN (Chen et al., ICCV 2021)
- HRI30 Dataset: https://zenodo.org/records/5833411
- NTU RGB+D Dataset: https://rose1.ntu.edu.sg/dataset/actionRecognition (ROSE Lab, NTU Singapore)

---

## Citation

```bibtex
@inproceedings{chowdhury2026robust,
  title     = {Robust Early Intent Prediction for Human-Robot Collaboration:
               A Multi-Model Consensus Approach under Partial Occlusion},
  author    = {Chowdhury, Soumitra and Alam, Mohammad Tausiful and Chowdhury, Antu},
  booktitle = {Proceedings of [Conference Name]},
  year      = {2026}
}
```

*BibTeX will be updated with full venue details after acceptance.*

---

## Contact

- Soumitra Chowdhury — 231000912@eastdelta.edu.bd
- Mohammad Tausiful Alam — 231007112@eastdelta.edu.bd
- Supervisor: Antu Chowdhury — antu.c@eastdelta.edu.bd
- Repository: https://github.com/Tausif7112/2.14.Mohammad-Tausiful-Alam_Soumitra-Chowdhury
