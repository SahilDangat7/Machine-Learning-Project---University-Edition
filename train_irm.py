"""
IRM + EfficientNet-B4 — Google Colab Version
PovertyMap-WILDS Dataset
Run on: Runtime → Change runtime type → T4 GPU
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ── Step 1: Install dependencies ───────────────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "wilds", "torchvision", "scipy"], check=True)

# ── Step 2: Imports ────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from scipy.stats import pearsonr
from wilds import get_dataset
import time
import os

# ── Step 3: Device ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Step 4: Reproducibility ────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Step 5: Hyperparameters ────────────────────────────────────────────────────
BATCH_SIZE    = 16
EPOCHS        = 50
LR            = 1e-3
PATIENCE      = 10
IRM_LAMBDA    = 1.0
IRM_ANNEAL    = 10
SAVE_PATH     = "/content/best_irm_model.pt"
N_GROUPS      = 2

# ── Step 6: Channel stats ──────────────────────────────────────────────────────
MEANS = [-0.0637, -0.0748, -0.0606, -0.0129, -0.0108, -0.0507, -0.0462,  0.2111]
STDS  = [ 0.9064,  0.9028,  0.9179,  0.9532,  0.9570,  0.9019,  0.9764,  1.1964]

# ── Step 7: Transforms ─────────────────────────────────────────────────────────
class NormalizeChannels:
    def __init__(self, means, stds):
        self.means = torch.tensor(means, dtype=torch.float32).view(8, 1, 1)
        self.stds  = torch.tensor(stds,  dtype=torch.float32).view(8, 1, 1)
    def __call__(self, x):
        return (x.float() - self.means) / (self.stds + 1e-6)

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

# ── Step 8: Data ───────────────────────────────────────────────────────────────
print("\nDownloading dataset (this may take a few minutes on Colab)...")
dataset = get_dataset(dataset="poverty", download=False)

train_data = dataset.get_subset("train", transform=TrainTransform(MEANS, STDS))
val_data   = dataset.get_subset("val",   transform=EvalTransform(MEANS, STDS))
test_data  = dataset.get_subset("test",  transform=EvalTransform(MEANS, STDS))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train batches : {len(train_loader)}")
print(f"Val batches   : {len(val_loader)}")
print(f"Test batches  : {len(test_loader)}")

# ── Step 9: Model ──────────────────────────────────────────────────────────────
class EfficientNetB4Poverty(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        old_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            8, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            for i in range(3, 8):
                new_conv.weight[:, i:i+1, :, :] = mean_weight
        self.backbone.features[0][0] = new_conv
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x)

model = EfficientNetB4Poverty().to(device)
print(f"\nModel: EfficientNet-B4 with 8-channel input")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Step 10: IRM penalty ───────────────────────────────────────────────────────
def irm_penalty(losses):
    dummy = torch.ones(losses.shape, requires_grad=True, device=losses.device)
    loss  = (losses * dummy).mean()
    grad  = torch.autograd.grad(loss, dummy, create_graph=True)[0]
    return (grad ** 2).mean()

# ── Step 11: Loss and optimiser ────────────────────────────────────────────────
criterion = nn.MSELoss(reduction="none")
optimiser = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

# ── Step 12: Evaluation ────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_domains = [], [], []
    total_loss = 0.0
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for x, y, meta in loader:
            x     = x.to(device)
            y     = y.float().to(device)
            preds = model(x).squeeze()
            loss  = loss_fn(preds, y.squeeze())
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.squeeze().cpu().numpy())
            all_domains.extend(meta[:, 0].numpy())

    avg_loss    = total_loss / len(loader)
    r_overall, _ = pearsonr(all_labels, all_preds)

    all_preds   = np.array(all_preds)
    all_labels  = np.array(all_labels)
    all_domains = np.array(all_domains)
    group_rs = {}
    for g in np.unique(all_domains):
        mask = all_domains == g
        if mask.sum() > 1:
            r_g, _ = pearsonr(all_labels[mask], all_preds[mask])
            group_rs[int(g)] = r_g

    worst_group_r = min(group_rs.values()) if group_rs else 0.0
    return avg_loss, r_overall, group_rs, worst_group_r

# ── Step 13: Training ──────────────────────────────────────────────────────────
print("\n── Training (IRM + EfficientNet-B4) ─────────────────")
print(f"{'Epoch':>6} {'Train Loss':>12} {'IRM Pen':>10} {'Val r':>8} {'Worst-g r':>10} {'Time':>8}")
print("-" * 62)

best_val_r     = -np.inf
patience_count = 0

for epoch in range(1, EPOCHS + 1):
    start = time.time()
    model.train()
    epoch_loss    = 0.0
    epoch_penalty = 0.0

    for batch_idx, (x, y, meta) in enumerate(train_loader):
        x       = x.to(device)
        y       = y.float().to(device)
        domains = meta[:, 0].long()

        optimiser.zero_grad()

        erm_loss  = torch.tensor(0.0, device=device)
        irm_pen   = torch.tensor(0.0, device=device)
        n_domains = 0

        for g in range(N_GROUPS):
            mask = domains == g
            if mask.sum() < 2:
                continue
            x_g    = x[mask]
            y_g    = y[mask].squeeze()
            preds  = model(x_g).squeeze()
            loss_g = criterion(preds, y_g)
            erm_loss = erm_loss + loss_g.mean()
            if epoch > IRM_ANNEAL:
                irm_pen = irm_pen + irm_penalty(loss_g)
            n_domains += 1

        if n_domains > 0:
            erm_loss = erm_loss / n_domains
            irm_pen  = irm_pen  / n_domains

        lam        = IRM_LAMBDA if epoch > IRM_ANNEAL else 0.0
        total_loss_batch = erm_loss + lam * irm_pen

        total_loss_batch.backward()
        optimiser.step()

        epoch_loss    += erm_loss.item()
        epoch_penalty += irm_pen.item()

        # Progress indicator
        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch} — batch {batch_idx+1}/{len(train_loader)}", end="\r")

    scheduler.step()
    train_loss = epoch_loss    / len(train_loader)
    train_pen  = epoch_penalty / len(train_loader)
    val_loss, val_r, group_rs, worst_r = evaluate(model, val_loader, device)
    elapsed = time.time() - start

    print(f"{epoch:>6} {train_loss:>12.4f} {train_pen:>10.4f} {val_r:>8.4f} {worst_r:>10.4f} {elapsed:>7.1f}s")
    if epoch <= IRM_ANNEAL:
        print(f"         [Warmup — IRM penalty starts epoch {IRM_ANNEAL+1}]")

    if val_r > best_val_r:
        best_val_r = val_r
        patience_count = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"         ↑ New best val r={val_r:.4f} — model saved")
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

# ── Step 14: Final test evaluation ─────────────────────────────────────────────
print("\n── Loading best model for test evaluation ────────────")
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
test_loss, test_r, test_group_rs, test_worst_r = evaluate(model, test_loader, device)

print(f"\n── IRM + EfficientNet-B4 Results ─────────────────────")
print(f"  Best Val Pearson r      : {best_val_r:.4f}")
print(f"  Test Pearson r          : {test_r:.4f}")
print(f"  Test MSE                : {test_loss:.4f}")
print(f"  Test Worst-group r      : {test_worst_r:.4f}")
for g, r in test_group_rs.items():
    print(f"  Domain {g} r             : {r:.4f}")

print(f"\n── Final Comparison ──────────────────────────────────")
print(f"  {'Method':<20} {'Model':<20} {'Test r':<10} {'Worst-g r'}")
print(f"  {'-'*60}")
print(f"  {'ERM':<20} {'ResNet-18':<20} {'0.8314':<10} {'?'}")
print(f"  {'IRM':<20} {'EfficientNet-B4':<20} {test_r:<10.4f} {test_worst_r:.4f}")

# ── Step 15: Download model ────────────────────────────────────────────────────
from google.colab import files
files.download(SAVE_PATH)
print("\nModel downloaded to your computer.")