"""
Stage 4 — ERM Baseline Training
ResNet-18 with 8-channel input, MSE loss, Adam optimiser
PovertyMap-WILDS Dataset
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from scipy.stats import pearsonr
from wilds import get_dataset
import time
import os

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ── Reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 1e-3
PATIENCE     = 10       # early stopping
SAVE_PATH    = "best_erm_model.pt"

# ── Transforms ─────────────────────────────────────────────────────────────────
class NormalizeChannels:
    def __init__(self, means, stds):
        self.means = torch.tensor(means, dtype=torch.float32).view(8, 1, 1)
        self.stds  = torch.tensor(stds,  dtype=torch.float32).view(8, 1, 1)
    def __call__(self, x):
        x = x.float()
        return (x - self.means) / (self.stds + 1e-6)

class TrainTransform:
    def __init__(self, means, stds):
        self.normalise = NormalizeChannels(means, stds)
    def __call__(self, x):
        x = x.float()
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[2])
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[1])
        k = torch.randint(0, 4, (1,)).item()
        x = torch.rot90(x, k, dims=[1, 2])
        return self.normalise(x)

class EvalTransform:
    def __init__(self, means, stds):
        self.normalise = NormalizeChannels(means, stds)
    def __call__(self, x):
        return self.normalise(x.float())

# Channel stats from data_prep.py
MEANS = [-0.0637, -0.0748, -0.0606, -0.0129, -0.0108, -0.0507, -0.0462,  0.2111]
STDS  = [ 0.9064,  0.9028,  0.9179,  0.9532,  0.9570,  0.9019,  0.9764,  1.1964]

# ── Data ────────────────────────────────────────────────────────────────────────
print("Loading data...")
dataset = get_dataset(dataset="poverty", download=False)

train_data = dataset.get_subset("train", transform=TrainTransform(MEANS, STDS))
val_data   = dataset.get_subset("val",   transform=EvalTransform(MEANS, STDS))
test_data  = dataset.get_subset("test",  transform=EvalTransform(MEANS, STDS))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train batches : {len(train_loader)}")
print(f"Val batches   : {len(val_loader)}")
print(f"Test batches  : {len(test_loader)}")

# ── Model ───────────────────────────────────────────────────────────────────────
class ResNet18Poverty(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet-18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Modify first conv to accept 8 channels instead of 3
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialise extra channels from mean of RGB weights
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            for i in range(3, 8):
                new_conv.weight[:, i:i+1, :, :] = mean_weight
        self.backbone.conv1 = new_conv
        # Replace final FC layer with regression head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)

model = ResNet18Poverty().to(device)
print(f"\nModel: ResNet-18 with 8-channel input")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ── Loss and optimiser ─────────────────────────────────────────────────────────
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

# ── Evaluation function ────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.float().to(device)
            preds = model(x).squeeze()
            loss  = criterion(preds, y.squeeze())
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.squeeze().cpu().numpy())
    avg_loss = total_loss / len(loader)
    r, _     = pearsonr(all_labels, all_preds)
    return avg_loss, r

# ── Training loop ──────────────────────────────────────────────────────────────
print("\n── Training ──────────────────────────────────────────")
print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val r':>8} {'Time':>8}")
print("-" * 52)

best_val_r    = -np.inf
patience_count = 0
train_losses, val_losses, val_rs = [], [], []

for epoch in range(1, EPOCHS + 1):
    start = time.time()
    model.train()
    epoch_loss = 0.0

    for x, y, _ in train_loader:
        x = x.to(device)
        y = y.float().to(device)
        optimiser.zero_grad()
        preds = model(x).squeeze()
        loss  = criterion(preds, y.squeeze())
        loss.backward()
        optimiser.step()
        epoch_loss += loss.item()

    scheduler.step()
    train_loss = epoch_loss / len(train_loader)
    val_loss, val_r = evaluate(model, val_loader, device)
    elapsed = time.time() - start

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_rs.append(val_r)

    print(f"{epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} {val_r:>8.4f} {elapsed:>7.1f}s")

    # Save best model
    if val_r > best_val_r:
        best_val_r = val_r
        patience_count = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"         ↑ New best val r={val_r:.4f} — model saved")
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

# ── Final evaluation on test set ───────────────────────────────────────────────
print("\n── Loading best model for test evaluation ────────────")
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
test_loss, test_r = evaluate(model, test_loader, device)

print(f"\n── ERM Baseline Results ──────────────────────────────")
print(f"  Best Val Pearson r  : {best_val_r:.4f}")
print(f"  Test MSE            : {test_loss:.4f}")
print(f"  Test Pearson r      : {test_r:.4f}")
print(f"\nModel saved to: {SAVE_PATH}")
print("Ready for Stage 5 — Group DRO")