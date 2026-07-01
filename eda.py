"""
Stage 2 — Exploratory Data Analysis
PovertyMap-WILDS Dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from wilds import get_dataset

# ── 1. Load dataset ────────────────────────────────────────────────────────────
print("Loading dataset...")
dataset = get_dataset(dataset="poverty", download=False)

train_data = dataset.get_subset("train")
val_data   = dataset.get_subset("val")
test_data  = dataset.get_subset("test")

print(f"Train : {len(train_data)} samples")
print(f"Val   : {len(val_data)} samples")
print(f"Test  : {len(test_data)} samples")

# ── 2. Extract labels and metadata ─────────────────────────────────────────────
print("\nExtracting labels and metadata...")

def extract_metadata(subset):
    labels, domains, urban = [], [], []
    for i in range(len(subset)):
        x, y, meta = subset[i]
        labels.append(float(y))
        domains.append(float(meta[0]))   # domain index
        urban.append(float(meta[3]))     # urban flag
    return np.array(labels), np.array(domains), np.array(urban)

train_labels, train_domains, train_urban = extract_metadata(train_data)
val_labels,   val_domains,   val_urban   = extract_metadata(val_data)
test_labels,  test_domains,  test_urban  = extract_metadata(test_data)

print("Done.")

# ── 3. Print statistics ────────────────────────────────────────────────────────
print("\n── Label Statistics ──────────────────────────────────")
for name, labels in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
    print(f"{name:6s} | mean={labels.mean():.3f}  std={labels.std():.3f}  "
          f"min={labels.min():.3f}  max={labels.max():.3f}")

print("\n── Urban/Rural Split ─────────────────────────────────")
for name, urban in [("Train", train_urban), ("Val", val_urban), ("Test", test_urban)]:
    n_urban = int((urban == 1).sum())
    n_rural = int((urban == 0).sum())
    print(f"{name:6s} | Urban={n_urban} ({100*n_urban/len(urban):.1f}%)  "
          f"Rural={n_rural} ({100*n_rural/len(urban):.1f}%)")

print("\n── Domain Split ──────────────────────────────────────")
for name, domains in [("Train", train_domains), ("Val", val_domains), ("Test", test_domains)]:
    print(f"{name:6s} | Domains: {np.unique(domains)}")

print("\n── Key OOD Finding ───────────────────────────────────")
print("Train is URBAN, Val/Test are RURAL → urban-to-rural generalisation problem")
print(f"Train mean wealth  : {train_labels.mean():.3f}")
print(f"Val   mean wealth  : {val_labels.mean():.3f}")
print(f"Test  mean wealth  : {test_labels.mean():.3f}")

# ── 4. Load sample images ──────────────────────────────────────────────────────
print("\nLoading sample images...")
x_poor, y_poor, _ = train_data[int(np.argmin(train_labels))]
x_rich, y_rich, _ = train_data[int(np.argmax(train_labels))]
x_mid,  y_mid,  _ = train_data[int(np.abs(train_labels - train_labels.mean()).argmin())]

y_poor = float(y_poor)
y_rich = float(y_rich)
y_mid  = float(y_mid)

# Also grab a rural sample from val
x_rural, y_rural, _ = val_data[0]
y_rural = float(y_rural)

BAND_NAMES = ["Coastal", "Blue", "Green", "Red", "NIR", "SWIR-1", "SWIR-2", "Nightlights"]

# ── 5. Plots ───────────────────────────────────────────────────────────────────
print("Generating plots...")
plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(20, 26))
fig.suptitle("PovertyMap-WILDS — Exploratory Data Analysis",
             fontsize=18, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(6, 4, figure=fig, hspace=0.5, wspace=0.35)

# Plot 1: Label distribution across splits
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(train_labels, bins=60, alpha=0.7, color="#2196F3", label=f"Train (urban, n={len(train_labels)})")
ax1.hist(val_labels,   bins=60, alpha=0.6, color="#FF9800", label=f"Val (rural, n={len(val_labels)})")
ax1.hist(test_labels,  bins=60, alpha=0.6, color="#4CAF50", label=f"Test (rural, n={len(test_labels)})")
ax1.axvline(train_labels.mean(), color="#2196F3", linestyle="--", linewidth=1.5)
ax1.axvline(val_labels.mean(),   color="#FF9800", linestyle="--", linewidth=1.5)
ax1.set_title("Wealth Index Distribution — Urban (Train) vs Rural (Val/Test)", fontweight="bold")
ax1.set_xlabel("Asset Wealth Index")
ax1.set_ylabel("Count")
ax1.legend()

# Plot 2: Box plot comparison urban vs rural
ax2 = fig.add_subplot(gs[0, 2:])
ax2.boxplot([train_labels, val_labels, test_labels],
            labels=["Train\n(urban)", "Val\n(rural)", "Test\n(rural)"],
            patch_artist=True,
            boxprops=dict(facecolor="#E3F2FD"),
            medianprops=dict(color="#1565C0", linewidth=2))
ax2.set_title("Wealth Distribution Boxplot", fontweight="bold")
ax2.set_ylabel("Asset Wealth Index")
ax2.axhline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

# Plot 3: Wealth histogram train only with mean lines
ax3 = fig.add_subplot(gs[1, :2])
ax3.hist(train_labels, bins=60, color="#673AB7", alpha=0.8)
ax3.axvline(train_labels.mean(), color="red",    linestyle="--", label=f"Mean={train_labels.mean():.2f}")
ax3.axvline(np.median(train_labels), color="orange", linestyle="-.", label=f"Median={np.median(train_labels):.2f}")
ax3.set_title("Train Wealth Distribution Detail", fontweight="bold")
ax3.set_xlabel("Asset Wealth Index")
ax3.set_ylabel("Count")
ax3.legend()

# Plot 4: Cumulative distribution
ax4 = fig.add_subplot(gs[1, 2:])
for labels, name, color in [(train_labels, "Train (urban)", "#2196F3"),
                              (val_labels,   "Val (rural)",   "#FF9800"),
                              (test_labels,  "Test (rural)",  "#4CAF50")]:
    sorted_l = np.sort(labels)
    cdf = np.arange(1, len(sorted_l)+1) / len(sorted_l)
    ax4.plot(sorted_l, cdf, label=name, color=color, linewidth=2)
ax4.set_title("Cumulative Distribution of Wealth", fontweight="bold")
ax4.set_xlabel("Asset Wealth Index")
ax4.set_ylabel("Cumulative Proportion")
ax4.legend()

# Plot 5: Domain counts
ax5 = fig.add_subplot(gs[2, :2])
domain_counts_train = [(d, int((train_domains == d).sum())) for d in np.unique(train_domains)]
ax5.bar([f"Domain {int(d)}" for d, _ in domain_counts_train],
        [c for _, c in domain_counts_train], color="#009688", alpha=0.8)
ax5.set_title("Samples per Domain (Train)", fontweight="bold")
ax5.set_ylabel("Count")

# Plot 6: Mean wealth per domain
ax6 = fig.add_subplot(gs[2, 2:])
all_domains = np.concatenate([train_domains, val_domains, test_domains])
all_labels  = np.concatenate([train_labels,  val_labels,  test_labels])
domain_means = [(d, all_labels[all_domains == d].mean()) for d in np.unique(all_domains)]
colors = ["#F44336" if v < 0 else "#4CAF50" for _, v in domain_means]
ax6.bar([f"Domain {int(d)}" for d, _ in domain_means],
        [v for _, v in domain_means], color=colors, alpha=0.8)
ax6.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax6.set_title("Mean Wealth per Domain (All Splits)", fontweight="bold")
ax6.set_ylabel("Mean Wealth Index")

# Plot 7: All 8 channels — poor urban area
for i in range(4):
    ax = fig.add_subplot(gs[3, i])
    ax.imshow(x_poor[i].numpy(), cmap="viridis")
    ax.set_title(f"{BAND_NAMES[i]}\npoor urban y={y_poor:.2f}", fontsize=9)
    ax.axis("off")

for i in range(4):
    ax = fig.add_subplot(gs[4, i])
    ax.imshow(x_poor[i+4].numpy(), cmap="viridis")
    ax.set_title(f"{BAND_NAMES[i+4]}\npoor urban y={y_poor:.2f}", fontsize=9)
    ax.axis("off")

# Plot 8: RGB comparison
def to_rgb(x):
    rgb = x[[3, 2, 1]].numpy()
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    return np.transpose(rgb, (1, 2, 0))

ax_poor = fig.add_subplot(gs[5, 0])
ax_poor.imshow(to_rgb(x_poor))
ax_poor.set_title(f"Poor urban\n{y_poor:.3f}", fontweight="bold", fontsize=10)
ax_poor.axis("off")

ax_mid = fig.add_subplot(gs[5, 1])
ax_mid.imshow(to_rgb(x_mid))
ax_mid.set_title(f"Average urban\n{y_mid:.3f}", fontweight="bold", fontsize=10)
ax_mid.axis("off")

ax_rich = fig.add_subplot(gs[5, 2])
ax_rich.imshow(to_rgb(x_rich))
ax_rich.set_title(f"Wealthy urban\n{y_rich:.3f}", fontweight="bold", fontsize=10)
ax_rich.axis("off")

# Plot 9: Nighttime lights poor vs rich
ax_nl = fig.add_subplot(gs[5, 3])
nl_poor  = float(x_poor[7].numpy().mean())
nl_rich  = float(x_rich[7].numpy().mean())
nl_rural = float(x_rural[7].numpy().mean())
ax_nl.bar(["Poor\nurban", "Rich\nurban", "Rural\nsample"],
          [nl_poor, nl_rich, nl_rural],
          color=["#F44336", "#4CAF50", "#FF9800"], alpha=0.85)
ax_nl.set_title("Avg Nighttime Light", fontweight="bold", fontsize=10)
ax_nl.set_ylabel("Mean intensity")

plt.savefig("eda_output.png", dpi=150, bbox_inches="tight")
print("\nSaved: eda_output.png")
print("\n── EDA Complete ──────────────────────────────────────────────────────")
print(f"  Wealth range     : {train_labels.min():.2f} to {train_labels.max():.2f}")
print(f"  Train mean       : {train_labels.mean():.3f} (urban)")
print(f"  Val mean         : {val_labels.mean():.3f} (rural)")
print(f"  Test mean        : {test_labels.mean():.3f} (rural)")
print(f"  Label shift      : {abs(train_labels.mean() - test_labels.mean()):.3f} gap train vs test")
print(f"  Train domains    : {np.unique(train_domains)}")
print(f"  Test domains     : {np.unique(test_domains)}")