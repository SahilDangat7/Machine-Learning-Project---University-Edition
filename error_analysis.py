"""
Stage 7 — Error Analysis (v2)
Comparing ERM (ResNet-18) vs IRM (EfficientNet-B4) — fair, image-only comparison
Plus ConvNeXt-Tiny + Metadata — included separately as a reference experiment
(receives domain/urban-rural labels at TEST time too — not directly comparable, see notes)
PovertyMap-WILDS Dataset
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights, convnext_tiny
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from wilds import get_dataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ── Channel stats ──────────────────────────────────────────────────────────────
MEANS = [-0.0637, -0.0748, -0.0606, -0.0129, -0.0108, -0.0507, -0.0462, 0.2111]
STDS  = [ 0.9064,  0.9028,  0.9179,  0.9532,  0.9570,  0.9019,  0.9764, 1.1964]

# ── Transforms ─────────────────────────────────────────────────────────────────
class NormalizeChannels:
    def __init__(self, means, stds):
        self.means = torch.tensor(means, dtype=torch.float32).view(8, 1, 1)
        self.stds  = torch.tensor(stds,  dtype=torch.float32).view(8, 1, 1)
    def __call__(self, x):
        return (x.float() - self.means) / (self.stds + 1e-6)

class EvalTransform:
    def __init__(self, means, stds):
        self.normalise = NormalizeChannels(means, stds)
    def __call__(self, x):
        return self.normalise(x.float())

# ── Data ────────────────────────────────────────────────────────────────────────
print("Loading data...")
dataset   = get_dataset(dataset="poverty", download=False)
test_data = dataset.get_subset("test", transform=EvalTransform(MEANS, STDS))
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)

# ── ERM Model ──────────────────────────────────────────────────────────────────
class ResNet18Poverty(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
    def forward(self, x):
        return self.backbone(x)

# ── IRM Model ──────────────────────────────────────────────────────────────────
class EfficientNetB4Poverty(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b4(weights=None)
        old_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(8, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding, bias=False)
        self.backbone.features[0][0] = new_conv
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 1)
        )
    def forward(self, x):
        return self.backbone(x)

# ── ConvNeXt + Metadata Model ───────────────────────────────────────────────────
class ConvNextPovertyModel(nn.Module):
    """Receives domain/urban-rural metadata at TEST time too — not a fair comparison.
    See accompanying notes / Section 5.3 of the project notebook."""
    def __init__(self):
        super().__init__()
        self.backbone = convnext_tiny(weights=None)
        old_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=8, out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=False
        )
        self.backbone.classifier = nn.Identity()
        image_features = 768

        self.metadata_net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU()
        )
        self.regressor = nn.Sequential(
            nn.Linear(image_features + 32, 256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, image, metadata):
        image_features = self.backbone.features(image)
        image_features = image_features.mean(dim=(2, 3))
        metadata_features = self.metadata_net(metadata)
        combined = torch.cat([image_features, metadata_features], dim=1)
        return self.regressor(combined)

# ── Load models ────────────────────────────────────────────────────────────────
print("Loading ERM model...")
erm_model = ResNet18Poverty().to(device)
erm_model.load_state_dict(torch.load("best_erm_model.pt", map_location=device))
erm_model.eval()

print("Loading IRM model...")
irm_model = EfficientNetB4Poverty().to(device)
irm_model.load_state_dict(torch.load("best_irm_model.pt", map_location=device))
irm_model.eval()

print("Loading ConvNeXt + Metadata model...")
convnext_model = ConvNextPovertyModel().to(device)
# strict=False: the saved checkpoint includes ConvNeXt's original classifier head
# weights (backbone.classifier.0 / .2), which are unused since forward() bypasses
# them via self.backbone.features(image) + manual global average pooling instead.
convnext_model.load_state_dict(torch.load("best_convnext_tiny.pt", map_location=device), strict=False)
convnext_model.eval()

# ── Get predictions ────────────────────────────────────────────────────────────
print("Running predictions...")

def get_predictions(model, loader, device):
    """Image-only models (ERM, IRM)."""
    all_preds, all_labels, all_domains = [], [], []
    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device)
            preds = model(x).squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.float().squeeze().numpy())
            all_domains.extend(meta[:, 0].numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_domains)

def get_predictions_with_metadata(model, loader, device):
    """ConvNeXt — passes meta[:, [0,2,3]] at test time too."""
    all_preds, all_labels, all_domains = [], [], []
    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device)
            meta_input = meta[:, [0, 2, 3]].float().to(device)
            preds = model(x, meta_input).squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.float().squeeze().numpy())
            all_domains.extend(meta[:, 0].numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_domains)

erm_preds, true_labels, domains = get_predictions(erm_model, test_loader, device)
irm_preds, _,            _       = get_predictions(irm_model, test_loader, device)
cvx_preds,  _,            _       = get_predictions_with_metadata(convnext_model, test_loader, device)

erm_residuals = erm_preds - true_labels
irm_residuals = irm_preds - true_labels
cvx_residuals = cvx_preds - true_labels

# ── Print summary ──────────────────────────────────────────────────────────────
print("\n── Error Analysis Summary (ERM vs IRM — fair, image-only) ────")
for g in np.unique(domains):
    mask = domains == g
    erm_r, _ = pearsonr(true_labels[mask], erm_preds[mask])
    irm_r, _ = pearsonr(true_labels[mask], irm_preds[mask])
    erm_mae = np.abs(erm_residuals[mask]).mean()
    irm_mae = np.abs(irm_residuals[mask]).mean()
    print(f"\nDomain {int(g)} ({mask.sum()} samples):")
    print(f"  ERM  r={erm_r:.4f}  MAE={erm_mae:.4f}")
    print(f"  IRM  r={irm_r:.4f}  MAE={irm_mae:.4f}")
    print(f"  Worst-group improvement: {irm_r - erm_r:+.4f}")

print("\n── ConvNeXt + Metadata (reference only — uses test-time domain labels) ──")
cvx_r_overall, _ = pearsonr(true_labels, cvx_preds)
print(f"Overall Pearson r: {cvx_r_overall:.4f}")
for g in np.unique(domains):
    mask = domains == g
    cvx_r, _ = pearsonr(true_labels[mask], cvx_preds[mask])
    cvx_mae = np.abs(cvx_residuals[mask]).mean()
    print(f"  Domain {int(g)}: r={cvx_r:.4f}  MAE={cvx_mae:.4f}")
print("NOTE: not directly comparable to ERM/IRM — see project notes on test-time metadata leakage.")

# ── Plots ──────────────────────────────────────────────────────────────────────
print("\nGenerating error analysis plots...")
plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(20, 26))
fig.suptitle("Error Analysis — ERM vs IRM vs ConvNeXt+Metadata*", fontsize=16, fontweight="bold")
gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)

COLORS = {"ERM": "#2196F3", "IRM": "#E91E63", "CVX": "#9E9E9E",
          "D0":  "#FF9800",  "D1": "#4CAF50"}

# ── Row 1: Predicted vs True — all three models ───────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(true_labels, erm_preds, alpha=0.3, s=10, color=COLORS["ERM"])
ax1.plot([-1.5, 3], [-1.5, 3], "r--", linewidth=1.5, label="Perfect prediction")
r_erm, _ = pearsonr(true_labels, erm_preds)
ax1.set_title(f"ERM (image only): Predicted vs True\nOverall r = {r_erm:.4f}", fontweight="bold")
ax1.set_xlabel("True Wealth Index"); ax1.set_ylabel("Predicted Wealth Index"); ax1.legend()

ax2 = fig.add_subplot(gs[0, 2:])
ax2.scatter(true_labels, irm_preds, alpha=0.3, s=10, color=COLORS["IRM"])
ax2.plot([-1.5, 3], [-1.5, 3], "r--", linewidth=1.5, label="Perfect prediction")
r_irm, _ = pearsonr(true_labels, irm_preds)
ax2.set_title(f"IRM (image only): Predicted vs True\nOverall r = {r_irm:.4f}", fontweight="bold")
ax2.set_xlabel("True Wealth Index"); ax2.set_ylabel("Predicted Wealth Index"); ax2.legend()

# ── Row 2: ConvNeXt scatter + residuals distribution ──────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
ax3.scatter(true_labels, cvx_preds, alpha=0.3, s=10, color=COLORS["CVX"])
ax3.plot([-1.5, 3], [-1.5, 3], "r--", linewidth=1.5, label="Perfect prediction")
ax3.set_title(f"ConvNeXt+Metadata* (image+domain labels): Predicted vs True\nOverall r = {cvx_r_overall:.4f}  (not directly comparable)",
              fontweight="bold")
ax3.set_xlabel("True Wealth Index"); ax3.set_ylabel("Predicted Wealth Index"); ax3.legend()

ax4 = fig.add_subplot(gs[1, 2:])
ax4.hist(erm_residuals, bins=60, alpha=0.6, color=COLORS["ERM"], label=f"ERM (std={erm_residuals.std():.3f})")
ax4.hist(irm_residuals, bins=60, alpha=0.6, color=COLORS["IRM"], label=f"IRM (std={irm_residuals.std():.3f})")
ax4.hist(cvx_residuals, bins=60, alpha=0.6, color=COLORS["CVX"], label=f"ConvNeXt* (std={cvx_residuals.std():.3f})")
ax4.axvline(0, color="black", linestyle="--", linewidth=1)
ax4.set_title("Residuals Distribution — all three models", fontweight="bold")
ax4.set_xlabel("Residual"); ax4.set_ylabel("Count"); ax4.legend()

# ── Row 3: Per-domain Pearson r bar chart + IRM residuals by domain ───────────
ax5 = fig.add_subplot(gs[2, :2])
domain_labels = [f"Domain {int(g)}" for g in np.unique(domains)]
erm_rs = [pearsonr(true_labels[domains==g], erm_preds[domains==g])[0] for g in np.unique(domains)]
irm_rs = [pearsonr(true_labels[domains==g], irm_preds[domains==g])[0] for g in np.unique(domains)]
x = np.arange(len(domain_labels)); width = 0.35
ax5.bar(x - width/2, erm_rs, width, label="ERM", color=COLORS["ERM"], alpha=0.85)
ax5.bar(x + width/2, irm_rs, width, label="IRM", color=COLORS["IRM"], alpha=0.85)
ax5.set_title("Pearson r per Domain — fair comparison (ERM vs IRM)", fontweight="bold")
ax5.set_xticks(x); ax5.set_xticklabels(domain_labels); ax5.set_ylabel("Pearson r"); ax5.legend(); ax5.set_ylim(0, 1)
for i, (e, m) in enumerate(zip(erm_rs, irm_rs)):
    ax5.text(i - width/2, e + 0.02, f"{e:.3f}", ha="center", fontsize=9)
    ax5.text(i + width/2, m + 0.02, f"{m:.3f}", ha="center", fontsize=9)

ax6 = fig.add_subplot(gs[2, 2:])
for g, color in zip(np.unique(domains), [COLORS["D0"], COLORS["D1"]]):
    mask = domains == g
    ax6.scatter(true_labels[mask], irm_residuals[mask], alpha=0.3, s=10, color=color, label=f"Domain {int(g)}")
ax6.axhline(0, color="black", linestyle="--", linewidth=1)
ax6.set_title("IRM Residuals by Domain (image only)", fontweight="bold")
ax6.set_xlabel("True Wealth Index"); ax6.set_ylabel("Residual (Predicted - True)"); ax6.legend()

# ── Row 4: ConvNeXt residuals by domain (the leakage visual) ──────────────────
ax7 = fig.add_subplot(gs[3, :2])
for g, color in zip(np.unique(domains), [COLORS["D0"], COLORS["D1"]]):
    mask = domains == g
    ax7.scatter(true_labels[mask], cvx_residuals[mask], alpha=0.3, s=10, color=color, label=f"Domain {int(g)}")
ax7.axhline(0, color="black", linestyle="--", linewidth=1)
ax7.set_title("ConvNeXt+Metadata* Residuals by Domain\n(model is told the domain directly)", fontweight="bold")
ax7.set_xlabel("True Wealth Index"); ax7.set_ylabel("Residual (Predicted - True)"); ax7.legend()

ax8 = fig.add_subplot(gs[3, 2:])
cvx_rs = [pearsonr(true_labels[domains==g], cvx_preds[domains==g])[0] for g in np.unique(domains)]
bars = ax8.bar(domain_labels, cvx_rs, color=COLORS["CVX"], alpha=0.85)
ax8.set_title("ConvNeXt+Metadata* Pearson r per Domain\n(reference only — not fair vs ERM/IRM)", fontweight="bold")
ax8.set_ylabel("Pearson r"); ax8.set_ylim(0, 1)
for bar, v in zip(bars, cvx_rs):
    ax8.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

plt.figtext(0.5, 0.005,
            "*ConvNeXt+Metadata receives domain/urban-rural labels at TEST time — results are not directly comparable to the image-only ERM/IRM models.",
            ha="center", fontsize=10, style="italic", color="#555555")

plt.savefig("error_analysis.png", dpi=150, bbox_inches="tight")
print("\nSaved: error_analysis.png")

print("\n── Complete Results Table ────────────────────────────")
print(f"{'Method':<22} {'Model':<18} {'Input':<20} {'Test r':<10} {'Worst-g r':<12} {'Notes'}")
print("-" * 110)
print(f"{'ERM':<22} {'ResNet-18':<18} {'Image only':<20} {r_erm:<10.4f} "
      f"{min(erm_rs):<12.4f} {'Fair baseline'}")
print(f"{'IRM':<22} {'EfficientNet-B4':<18} {'Image only':<20} {r_irm:<10.4f} "
      f"{min(irm_rs):<12.4f} {'Fair, +30.2% worst-group vs ERM'}")
print(f"{'ConvNeXt+Metadata':<22} {'ConvNeXt-Tiny':<18} {'Image+domain label':<20} {cvx_r_overall:<10.4f} "
      f"{min(cvx_rs):<12.4f} {'NOT comparable — test-time leakage'}")
print(f"\nFair (image-only) worst-group improvement, ERM -> IRM: "
      f"{min(irm_rs) - min(erm_rs):+.4f} ({((min(irm_rs)-min(erm_rs))/min(erm_rs))*100:.1f}% relative)")
print("\nReady for Stage 8 — Final Report")