"""
Stage 3 — Data Preparation
PovertyMap-WILDS Dataset
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from wilds import get_dataset

# ── 1. Load dataset ────────────────────────────────────────────────────────────
print("Loading dataset...")
dataset = get_dataset(dataset="poverty", download=False)

# ── 2. Compute per-channel mean and std from training data ─────────────────────
# We need to normalise all 8 channels independently
# These are approximate values for the PovertyMap dataset

print("Computing channel statistics...")

def compute_channel_stats(subset, n_samples=500):
    """Compute mean and std for each of the 8 channels."""
    all_pixels = [[] for _ in range(8)]
    indices = np.random.choice(len(subset), size=min(n_samples, len(subset)), replace=False)
    for i in indices:
        x, _, _ = subset[i]
        for c in range(8):
            all_pixels[c].append(x[c].numpy().flatten())
    means = [np.concatenate(all_pixels[c]).mean() for c in range(8)]
    stds  = [np.concatenate(all_pixels[c]).std()  for c in range(8)]
    return means, stds

train_data_raw = dataset.get_subset("train")
means, stds = compute_channel_stats(train_data_raw, n_samples=500)

print("\nChannel statistics (from 500 training samples):")
BAND_NAMES = ["Coastal", "Blue", "Green", "Red", "NIR", "SWIR-1", "SWIR-2", "Nightlights"]
for i, (m, s) in enumerate(zip(means, stds)):
    print(f"  {BAND_NAMES[i]:12s} | mean={m:.4f}  std={s:.4f}")

# ── 3. Define transforms ───────────────────────────────────────────────────────

class NormalizeChannels:
    """Normalise each of the 8 channels with its own mean/std."""
    def __init__(self, means, stds):
        self.means = torch.tensor(means, dtype=torch.float32).view(8, 1, 1)
        self.stds  = torch.tensor(stds,  dtype=torch.float32).view(8, 1, 1)

    def __call__(self, x):
        x = x.float()
        return (x - self.means) / (self.stds + 1e-6)


class TrainTransform:
    """Augmentations for training — applied to the full 8-channel tensor."""
    def __init__(self, means, stds):
        self.normalise = NormalizeChannels(means, stds)

    def __call__(self, x):
        x = x.float()
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[2])
        # Random vertical flip
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[1])
        # Random 90-degree rotation
        k = torch.randint(0, 4, (1,)).item()
        x = torch.rot90(x, k, dims=[1, 2])
        # Normalise
        x = self.normalise(x)
        return x


class EvalTransform:
    """No augmentation for val/test — just normalise."""
    def __init__(self, means, stds):
        self.normalise = NormalizeChannels(means, stds)

    def __call__(self, x):
        return self.normalise(x.float())


# ── 4. Create datasets with transforms ────────────────────────────────────────
train_transform = TrainTransform(means, stds)
eval_transform  = EvalTransform(means, stds)

train_data = dataset.get_subset("train", transform=train_transform)
val_data   = dataset.get_subset("val",   transform=eval_transform)
test_data  = dataset.get_subset("test",  transform=eval_transform)

# ── 5. Create DataLoaders ──────────────────────────────────────────────────────
BATCH_SIZE  = 64
NUM_WORKERS = 0   # set to 4 if you have a powerful machine

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

# ── 6. Verify one batch ────────────────────────────────────────────────────────
print("\nVerifying DataLoaders...")
x_batch, y_batch, meta_batch = next(iter(train_loader))

print(f"\nTrain batch:")
print(f"  Images shape : {x_batch.shape}")
print(f"  Labels shape : {y_batch.shape}")
print(f"  Labels range : {y_batch.min().item():.3f} to {y_batch.max().item():.3f}")
print(f"  Meta shape   : {meta_batch.shape}")
print(f"  Image mean   : {x_batch.mean().item():.4f}  (should be near 0)")
print(f"  Image std    : {x_batch.std().item():.4f}   (should be near 1)")

print(f"\nVal batch:")
x_val, y_val, _ = next(iter(val_loader))
print(f"  Images shape : {x_val.shape}")
print(f"  Labels shape : {y_val.shape}")

print(f"\nTest batch:")
x_test, y_test, _ = next(iter(test_loader))
print(f"  Images shape : {x_test.shape}")
print(f"  Labels shape : {y_test.shape}")

print("\n── Data Preparation Complete ─────────────────────────────")
print(f"  Batch size        : {BATCH_SIZE}")
print(f"  Train batches     : {len(train_loader)}")
print(f"  Val batches       : {len(val_loader)}")
print(f"  Test batches      : {len(test_loader)}")
print(f"  Input shape       : [B, 8, 224, 224]")
print(f"  Augmentations     : H-flip, V-flip, 90-degree rotation")
print(f"  Normalisation     : per-channel mean/std")
print("\nReady for Stage 4 — ERM Baseline Training")