# Out-of-Distribution Generalisation for Poverty Mapping

**Course:** Machine Learning — Summer 2026  
**University:** University of Europe for Applied Sciences, Berlin  
**Instructor:** Prof. Iftikhar Ahmed  

---

## Project Overview

This project investigates out-of-distribution (OOD) generalisation for satellite-based poverty mapping using the **PovertyMap-WILDS** benchmark. Models are trained on urban satellite imagery and evaluated on rural imagery — a real distribution shift. We compare three approaches:

| Method | Model | Input | Test Pearson r | Worst-group r |
|--------|-------|-------|---------------|---------------|
| ERM (baseline) | ResNet-18 | Image only | 0.8314 | 0.3605 |
| IRM (robust) | EfficientNet-B4 | Image only | 0.8169 | **0.4693** |
| ConvNeXt + Metadata* | ConvNeXt-Tiny | Image + domain labels | 0.8601 | 0.5589 |

*\*ConvNeXt receives domain/urban-rural labels at test time — not directly comparable to the image-only models. See paper for discussion.*

**Key finding:** IRM improves worst-group Pearson r by **30.2% relative** over ERM, demonstrating more equitable performance across domains at a small cost to average accuracy.

---

## Dataset

**PovertyMap-WILDS** — automatically downloaded via the WILDS library.

- 9,797 training samples (urban), 3,909 validation + 3,963 test samples (rural)
- Input: 8-channel satellite image (Coastal, Blue, Green, Red, NIR, SWIR-1, SWIR-2, Nighttime Lights), 224×224 pixels
- Label: continuous asset wealth index
- Source: https://wilds.stanford.edu/datasets/

---

## Repository Structure

```
├── data/                        # Dataset cache (downloaded automatically, not committed)
├── best_erm_model.pt            # Saved ERM model weights
├── best_irm_model.pt            # Saved IRM model weights
├── best_convnext_tiny.pt        # Saved ConvNeXt model weights
├── eda.py                       # Stage 2: Exploratory data analysis
├── data_prep.py                 # Stage 3: Data preparation and DataLoader verification
├── train_erm.py                 # Stage 4: ERM baseline training (ResNet-18)
├── train_irm.py                 # Stage 5: IRM training (EfficientNet-B4), local version
├── train_irm_colab.py           # Stage 5: IRM training (EfficientNet-B4), Colab version
├── error_analysis_v2.py         # Stage 7: Error analysis for all three models
├── Poverty_Mapping_Project.ipynb # Main presentation notebook
├── eda_output.png               # EDA figure
├── error_analysis.png           # Error analysis figure
└── README.md
```

---

## Setup

### Requirements

```bash
pip install wilds torch torchvision scipy matplotlib numpy pandas scikit-learn
```

### Environment

- Python 3.10+
- PyTorch 2.x
- Tested on: Apple M1 (MPS) for ERM, Google Colab T4 GPU for IRM

---

## How to Run

### 1. Download the dataset

```python
from wilds import get_dataset
dataset = get_dataset(dataset="poverty", download=True)
```

> **Note:** If the WILDS SSL certificate is expired (common on Colab), use the manual download in `train_irm_colab.py` with `verify=False`.

### 2. Run EDA

```bash
python eda.py
```

Produces `eda_output.png` — label distributions, channel visualisations, urban/rural split analysis.

### 3. Data preparation check

```bash
python data_prep.py
```

Verifies DataLoaders, computes channel statistics, confirms normalisation works correctly.

### 4. Train ERM baseline (ResNet-18)

```bash
python train_erm.py
```

Trains on MPS (Mac) or CPU. Saves `best_erm_model.pt`. Takes ~2 hours on M1.

### 5. Train IRM model (EfficientNet-B4)

**On Google Colab (recommended — requires T4 GPU):**

Upload `train_irm_colab.py` to Colab and run:
```python
exec(open('train_irm_colab.py').read())
```

Saves `best_irm_model.pt` and downloads it automatically.

### 6. Error analysis

```bash
python error_analysis_v2.py
```

Loads all three saved models, runs predictions on the test set, produces `error_analysis.png`.

### 7. Presentation notebook

Open `Poverty_Mapping_Project.ipynb` in Colab or Jupyter. Run all cells top to bottom. Models load from Google Drive — no retraining needed.

---

## Model Architectures

### ERM — ResNet-18
- Standard supervised training, MSE loss averaged uniformly across all samples
- First conv layer modified to accept 8 satellite channels
- Extra channels initialised from mean of pretrained RGB weights

### IRM — EfficientNet-B4
- Invariant Risk Minimisation with IRM penalty weight λ = 1.0
- 10-epoch ERM warmup before penalty activates
- Batch split by domain during training to compute per-domain losses
- Domain labels used only to structure training loss, never fed as input features

### ConvNeXt-Tiny + Metadata
- Two-branch architecture: image branch (ConvNeXt-Tiny) + metadata branch (3→16→32 MLP)
- Metadata: domain id, urban-type field, binary urban/rural flag
- Concatenated before final regression head (800→256→64→1)
- **Important:** metadata is provided at test time too — this model answers a different question from ERM/IRM

---

## Key Results

```
Method          Model              Test r    D0 r    D1 r    Worst-group r
------------------------------------------------------------------------
ERM             ResNet-18          0.8314    0.3605  0.6963  0.3605
IRM             EfficientNet-B4    0.8169    0.4693  0.6444  0.4693  ← +30.2%
ConvNeXt+Meta*  ConvNeXt-Tiny      0.8601    0.5589  0.6675  0.5589

*Not directly comparable — receives domain labels at test time
```

---

## Reproducibility

- Random seeds fixed: `torch.manual_seed(42)`, `np.random.seed(42)`
- Channel normalisation statistics hardcoded from a one-time computation:
  - MEANS: `[-0.0637, -0.0748, -0.0606, -0.0129, -0.0108, -0.0507, -0.0462, 0.2111]`
  - STDS: `[0.9064, 0.9028, 0.9179, 0.9532, 0.9570, 0.9019, 0.9764, 1.1964]`
- All models use the official WILDS train/val/test splits without modification

---

## References

- Yeh et al. (2020). Using publicly available satellite imagery and deep learning to understand economic well-being in Africa. *Nature Communications*.
- Koh et al. (2021). WILDS: A Benchmark of in-the-Wild Distribution Shifts. *ICML*.
- Arjovsky et al. (2019). Invariant Risk Minimisation. *arXiv:1907.02893*.
- He et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Tan & Le (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML*.
- Liu et al. (2022). A ConvNet for the 2020s. *CVPR*.
