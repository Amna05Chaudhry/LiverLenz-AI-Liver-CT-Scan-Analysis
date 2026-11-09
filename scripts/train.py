"""Retrain the liver segmentation model on CPU (LiverLenz v2).

Fixes the problems found in the original notebook (see docs/AUDIT.md):

* masks are binarised (liver = 1) instead of feeding LiTS labels {0, 1, 2} to a sigmoid;
* the loss is BCE + soft Dice over the whole image (not per-pixel Dice);
* the split is **by patient**, so no slice of a test patient is seen in training;
* the model is selected on validation patients and evaluated once on test patients.

Data: CHAOS CT (20 patients, see scripts/chaos_data.py). Input pipeline is the
same as the app: 3-channel windowed slice resized to 128 x 128.

    python scripts/chaos_data.py
    python scripts/train.py            # ~40 min on an 8-core CPU
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from chaos_data import PATIENTS, load  # noqa: E402

SPLIT_SEED = 0
N_TEST, N_VAL = 5, 2


def split_patients(seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    p = list(rng.permutation(PATIENTS))
    return sorted(map(int, p[N_TEST + N_VAL:])), sorted(map(int, p[N_TEST:N_TEST + N_VAL])), sorted(map(int, p[:N_TEST]))


# ------------------------------------------------------------------ model
def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Compact 4-level U-Net (~1.1 M parameters with base=12)."""

    def __init__(self, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([block(3, c[0])] + [block(c[i], c[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([block(c[i] * 2, c[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):  # x: (N, 3, H, W) in [0, 1]
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.head(x)  # logits


class NHWCExport(nn.Module):
    """Same interface as the original Keras model: NHWC float in [0,1] -> NHWC probability."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.sigmoid(self.net(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)


# ------------------------------------------------------------------- data
def load_split(patients, empty_frac=1.0, seed=0):
    """Load slices; optionally keep only a fraction of liver-free slices (they are easy and slow training)."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for p in patients:
        d = load(p)
        keep = d["y"].any((1, 2)) | (rng.random(len(d["y"])) < empty_frac)
        xs.append(d["x"][keep]); ys.append(d["y"][keep])
    return np.concatenate(xs), np.concatenate(ys)


def augment(x: torch.Tensor, y: torch.Tensor):
    """Random flips (orientation robustness: LiTS/NIfTI exports are often flipped),
    90-degree rotations, and mild contrast/brightness jitter."""
    if torch.rand(1) < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if torch.rand(1) < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    k = int(torch.randint(0, 4, (1,)))
    if k and torch.rand(1) < 0.3:
        x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
    a = 1 + 0.15 * (torch.rand(x.shape[0], 1, 1, 1) - 0.5) * 2
    b = 0.05 * (torch.rand(x.shape[0], 1, 1, 1) - 0.5) * 2
    return (x * a + b).clamp(0, 1), y


def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + eps)).mean()


@torch.no_grad()
def evaluate(net, x, y, bs=64):
    net.eval()
    inter = tot = 0.0
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).permute(0, 3, 1, 2).float() / 255
        pred = (torch.sigmoid(net(xb)) > 0.5).squeeze(1).numpy()
        yb = y[i:i + bs].astype(bool)
        inter += np.logical_and(pred, yb).sum()
        tot += pred.sum() + yb.sum()
    return 2 * inter / max(tot, 1)  # global Dice over all slices of the split


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=12)
    ap.add_argument("--empty-frac", type=float, default=0.25, help="fraction of liver-free training slices kept")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "models" / "liverlenz_unet_v2"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    train_p, val_p, test_p = split_patients()
    print(f"train patients {train_p}\nval patients   {val_p}\ntest patients  {test_p} (not used here)", flush=True)
    xtr, ytr = load_split(train_p, args.empty_frac)
    xva, yva = load_split(val_p)
    print(f"train slices {len(xtr)} ({(ytr.any((1, 2))).mean():.0%} with liver), val slices {len(xva)}", flush=True)

    net = UNet(args.base)
    print(f"parameters: {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(xtr) / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)

    best, history = -1.0, []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        net.train()
        t0, losses = time.time(), []
        for idx in np.array_split(np.random.permutation(len(xtr)), int(np.ceil(len(xtr) / args.bs))):
            xb = torch.from_numpy(xtr[idx]).permute(0, 3, 1, 2).float() / 255
            yb = torch.from_numpy(ytr[idx]).unsqueeze(1).float()
            xb, yb = augment(xb, yb)
            logits = net(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb) + dice_loss(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
        vd = evaluate(net, xva, yva)
        history.append({"epoch": ep, "loss": float(np.mean(losses)), "val_dice": float(vd)})
        flag = ""
        if vd > best:
            best, flag = vd, "  *saved*"
            torch.save(net.state_dict(), out.with_suffix(".pt"))
        print(f"epoch {ep:2d}  loss {np.mean(losses):.4f}  val Dice {vd:.4f}  ({time.time() - t0:.0f}s){flag}", flush=True)

    net.load_state_dict(torch.load(out.with_suffix(".pt")))
    net.eval()
    torch.onnx.export(NHWCExport(net), torch.rand(1, 128, 128, 3), str(out.with_suffix(".onnx")),
                      input_names=["input"], output_names=["output"], opset_version=17,
                      dynamic_axes={"input": {0: "N"}, "output": {0: "N"}}, dynamo=False)
    meta = {"architecture": f"U-Net base {args.base}", "params_M": sum(p.numel() for p in net.parameters()) / 1e6,
            "train_patients": train_p, "val_patients": val_p, "test_patients": test_p,
            "best_val_dice": best, "epochs": args.epochs, "history": history, "args": vars(args)}
    (ROOT / "docs" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "results" / "training_v2.json").write_text(json.dumps(meta, indent=2))
    print(f"best val Dice {best:.4f}; exported {out.with_suffix('.onnx')}")


if __name__ == "__main__":
    main()
