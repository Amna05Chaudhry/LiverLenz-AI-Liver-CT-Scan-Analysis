# %% [markdown]
# # LiverLenz AI v2: liver + tumour segmentation on LiTS (Kaggle GPU)
#
# Two-stage cascade trained on the **LiTS** dataset (131 contrast-enhanced CT volumes, liver and tumour masks):
#
# 1. **Stage 1, liver**: U-Net (ResNet-18 encoder, ImageNet weights) on the full axial slice, 2.5-D input
#    (slice and its two neighbours as channels), 256 x 256.
# 2. **Stage 2, tumour**: U-Net (same design) on a crop around the liver, taken from the 512 x 512 slice and
#    resized to 256 x 256, so small lesions keep their detail.
#
# Fixes compared with the original notebook: binarised targets per stage, Dice + BCE/focal losses,
# **patient-wise** train/val/test split, best checkpoint chosen on validation patients, and 3-D evaluation
# on held-out test patients (liver Dice/IoU, tumour Dice/IoU, lesion-wise detection).
#
# Outputs (in `/kaggle/working/`): `liver_stage1.onnx`, `tumour_stage2.onnx`, `model_meta.json`,
# `metrics_test.json`, `split.json`, training curves and qualitative examples.
#
# **Kaggle setup**: add the datasets `andrewmvd/liver-tumor-segmentation` and
# `andrewmvd/liver-tumor-segmentation-part-2`, accelerator **GPU T4 x2** or **P100**, internet **on**.

# %%
import os, sys, subprocess
SMOKE = os.environ.get("LIVERLENZ_SMOKE") == "1"   # tiny local test on synthetic data
if not SMOKE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.3.4",
                    "onnx", "onnxruntime"], check=False)

import json, glob, math, random, time, gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from scipy import ndimage

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_ROOT = Path(os.environ.get("LITS_ROOT", "/kaggle/input"))
WORK = Path(os.environ.get("WORK_DIR", "/kaggle/working"))
CACHE = Path(os.environ.get("CACHE_DIR", "/kaggle/tmp/lits_cache"))
WORK.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)

# Preprocessing (shared with the app: liverlenz/cascade.py must match)
HU_MIN, HU_MAX = -100.0, 250.0          # abdomen/liver window
S1_SIZE, S2_SIZE = 256, 256
CROP_MARGIN = 16                         # px at 512 resolution around the liver box
Z_MARGIN = 12                            # slices kept above/below the liver
ENCODER = "resnet18"
EPOCHS_S1 = 2 if SMOKE else 14
EPOCHS_S2 = 2 if SMOKE else 24
SAMPLES_PER_EPOCH = 64 if SMOKE else 12000
BATCH = 4 if SMOKE else 32
print("device", DEVICE, "smoke", SMOKE)

# %% [markdown]
# ## 1. Find volumes, fix orientation, cache windowed slices
# Volumes are reoriented to RAS with nibabel and shown in **radiological display** (patient's right on the
# image left, anterior at the top), the same orientation as DICOM, so the app can take DICOM or NIfTI.

# %%
def find_pairs(root):
    vols, segs = {}, {}
    for p in root.rglob("*.nii*"):
        n = p.name
        if n.startswith("volume-"):
            vols[int(n.split("-")[1].split(".")[0])] = p
        elif n.startswith("segmentation-"):
            segs[int(n.split("-")[1].split(".")[0])] = p
    ids = sorted(set(vols) & set(segs))
    return {i: (vols[i], segs[i]) for i in ids}

def to_display(a):
    """RAS array (x, y, z) -> radiological display (rows = anterior->posterior, cols = patient right->left)."""
    return np.ascontiguousarray(a[::-1, ::-1].transpose(1, 0, 2))

def window_u8(hu):
    return ((np.clip(hu, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)

def cache_volume(vid, vpath, spath):
    out = CACHE / f"v{vid:03d}.npz"
    if out.exists():
        return vid
    v = nib.as_closest_canonical(nib.load(str(vpath)))
    s = nib.as_closest_canonical(nib.load(str(spath)))
    img = to_display(np.asarray(v.dataobj, dtype=np.float32))
    seg = to_display(np.asarray(s.dataobj)).astype(np.uint8)
    zooms = v.header.get_zooms()
    has_liver = np.where(seg.reshape(-1, seg.shape[2]).max(0) > 0)[0]
    if len(has_liver) == 0:
        return None
    z0, z1 = max(0, has_liver[0] - Z_MARGIN), min(seg.shape[2], has_liver[-1] + Z_MARGIN + 1)
    img, seg = window_u8(img[:, :, z0:z1]), seg[:, :, z0:z1]
    if img.shape[:2] != (512, 512):
        img = np.stack([cv2.resize(img[..., k], (512, 512), interpolation=cv2.INTER_AREA) for k in range(img.shape[2])], -1)
        seg = np.stack([cv2.resize(seg[..., k], (512, 512), interpolation=cv2.INTER_NEAREST) for k in range(seg.shape[2])], -1)
    sx = float(zooms[0]) * v.shape[0] / 512.0
    np.savez(out, img=img.transpose(2, 0, 1), seg=seg.transpose(2, 0, 1),
             spacing=np.array([float(zooms[2]), sx, sx]), z0=z0)
    return vid

if SMOKE:   # synthetic volumes: an ellipse "liver" with a bright "tumour"
    syn = INPUT_ROOT / "synthetic"; syn.mkdir(parents=True, exist_ok=True)
    for vid in range(8):
        if (syn / f"volume-{vid}.nii").exists(): continue
        zz = 30; vol = np.full((512, 512, zz), -900, np.float32); seg = np.zeros((512, 512, zz), np.uint8)
        yy, xx = np.mgrid[:512, :512]
        for k in range(5, 25):
            body = (xx - 256) ** 2 / 200 ** 2 + (yy - 256) ** 2 / 150 ** 2 < 1; vol[..., k][body] = 40
            liv = (xx - 330) ** 2 / 80 ** 2 + (yy - 230) ** 2 / 60 ** 2 < 1; vol[..., k][liv] = 110; seg[..., k][liv] = 1
            tum = (xx - 340 - vid) ** 2 + (yy - 220) ** 2 < 15 ** 2
            if 10 < k < 18: vol[..., k][tum] = 30; seg[..., k][tum] = 2
        vol += np.random.normal(0, 10, vol.shape)
        nib.save(nib.Nifti1Image(vol, np.diag([0.8, 0.8, 2.5, 1])), str(syn / f"volume-{vid}.nii"))
        nib.save(nib.Nifti1Image(seg, np.diag([0.8, 0.8, 2.5, 1])), str(syn / f"segmentation-{vid}.nii"))

pairs = find_pairs(INPUT_ROOT)
print(f"found {len(pairs)} volume/segmentation pairs")
t0 = time.time()
with ThreadPoolExecutor(4) as ex:
    done = [r for r in ex.map(lambda kv: cache_volume(kv[0], *kv[1]), pairs.items()) if r is not None]
print(f"cached {len(done)} volumes in {time.time() - t0:.0f}s")

# %% [markdown]
# ## 2. Patient-wise split (fixed seed, saved to `split.json`)

# %%
ids = sorted(done)
rng = np.random.default_rng(SEED); perm = [int(i) for i in rng.permutation(ids)]
n_test = max(1, round(0.15 * len(ids))); n_val = max(1, round(0.08 * len(ids)))
split = {"test": sorted(perm[:n_test]), "val": sorted(perm[n_test:n_test + n_val]), "train": sorted(perm[n_test + n_val:])}
(WORK / "split.json").write_text(json.dumps(split, indent=1))
print({k: len(v) for k, v in split.items()})

def load(vid):
    d = np.load(CACHE / f"v{vid:03d}.npz")
    return d["img"], d["seg"], d["spacing"]

# %% [markdown]
# ## 3. Datasets (2.5-D), GPU augmentation

# %%
def stack25(img, k):
    n = len(img)
    return np.stack([img[max(k - 1, 0)], img[k], img[min(k + 1, n - 1)]], 0)

def liver_box(mask, margin=CROP_MARGIN):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    side = max(y1 - y0, x1 - x0) + 2 * margin            # square crop keeps the aspect ratio
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    side = int(min(max(side, 96), 512))
    y0 = int(np.clip(cy - side // 2, 0, 512 - side)); x0 = int(np.clip(cx - side // 2, 0, 512 - side))
    return y0, x0, side

def crop_resize(arr, box, size, interp):
    y0, x0, s = box
    c = arr[..., y0:y0 + s, x0:x0 + s]
    if c.ndim == 2:
        return cv2.resize(c, (size, size), interpolation=interp)
    return np.stack([cv2.resize(ch, (size, size), interpolation=interp) for ch in c], 0)

class SliceSet(torch.utils.data.Dataset):
    """stage=1: full slice -> liver mask. stage=2: liver crop -> tumour mask (GT liver box + jitter)."""
    def __init__(self, vids, stage, samples, train=True):
        self.stage, self.samples, self.train = stage, samples, train
        self.vols = {v: load(v)[:2] for v in vids}
        self.index = []
        for v, (img, seg) in self.vols.items():
            for k in range(len(seg)):
                liver, tumour = (seg[k] > 0).any(), (seg[k] == 2).any()
                if stage == 1:
                    self.index.append((v, k, 3.0 if liver else 1.0))
                elif liver:
                    self.index.append((v, k, 4.0 if tumour else 1.0))   # oversample tumour slices
        w = np.array([i[2] for i in self.index]); self.p = w / w.sum()

    def __len__(self):
        return self.samples if self.train else len(self.index)

    def __getitem__(self, i):
        v, k, _ = self.index[np.random.choice(len(self.index), p=self.p)] if self.train else self.index[i]
        img, seg = self.vols[v]
        x = stack25(img, k)
        if self.stage == 1:
            x = np.stack([cv2.resize(ch, (S1_SIZE, S1_SIZE), interpolation=cv2.INTER_AREA) for ch in x])
            y = cv2.resize((seg[k] > 0).astype(np.uint8), (S1_SIZE, S1_SIZE), interpolation=cv2.INTER_NEAREST)
        else:
            box = liver_box(seg[k] > 0, CROP_MARGIN + (np.random.randint(-8, 24) if self.train else 0))
            x = crop_resize(x, box, S2_SIZE, cv2.INTER_LINEAR)
            y = crop_resize((seg[k] == 2).astype(np.uint8), box, S2_SIZE, cv2.INTER_NEAREST)
        if self.train and np.random.rand() < 0.2:        # "2-D mode": single image uploads repeat one slice
            x = np.repeat(x[1:2], 3, 0)
        return torch.from_numpy(x).float() / 255.0, torch.from_numpy(y[None]).float()

def gpu_augment(x, y):
    b = x.shape[0]
    if torch.rand(1) < 0.5: x, y = x.flip(-1), y.flip(-1)     # orientation robustness
    if torch.rand(1) < 0.2: x, y = x.flip(-2), y.flip(-2)
    ang = (torch.rand(b, device=x.device) - 0.5) * 0.5        # +-14 degrees
    sc = 1 + (torch.rand(b, device=x.device) - 0.5) * 0.3
    tx = (torch.rand(b, 2, device=x.device) - 0.5) * 0.15
    th = torch.zeros(b, 2, 3, device=x.device)
    th[:, 0, 0] = torch.cos(ang) / sc; th[:, 0, 1] = -torch.sin(ang) / sc
    th[:, 1, 0] = torch.sin(ang) / sc; th[:, 1, 1] = torch.cos(ang) / sc; th[:, :, 2] = tx
    grid = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    y = (F.grid_sample(y, grid, mode="nearest", align_corners=False) > 0.5).float()
    a = 1 + (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 0.3
    c = (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 0.1
    return (x * a + c).clamp(0, 1), y

# %% [markdown]
# ## 4. Training loop (AMP, cosine LR, best checkpoint on validation Dice)

# %%
def make_model():
    return smp.Unet(ENCODER, encoder_weights=None if SMOKE else "imagenet", in_channels=3, classes=1)

def soft_dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits); inter = (p * y).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + eps)).mean()

def focal_loss(logits, y, gamma=2.0, alpha=0.75):
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.sigmoid(logits); pt = torch.where(y > 0.5, p, 1 - p)
    w = torch.where(y > 0.5, alpha, 1 - alpha)
    return (w * (1 - pt) ** gamma * bce).mean()

@torch.no_grad()
def val_dice(model, loader):
    model.eval(); inter = tot = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
            p = (torch.sigmoid(model(x)) > 0.5).float()
        inter += (p * y).sum().item(); tot += p.sum().item() + y.sum().item()
    return 2 * inter / max(tot, 1.0)

def train_stage(stage, epochs):
    tr = SliceSet(split["train"], stage, SAMPLES_PER_EPOCH, train=True)
    va = SliceSet(split["val"], stage, 0, train=False)
    tl = torch.utils.data.DataLoader(tr, batch_size=BATCH, num_workers=0 if SMOKE else 4, drop_last=True)
    vl = torch.utils.data.DataLoader(va, batch_size=BATCH, num_workers=0 if SMOKE else 4)
    model = make_model().to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * len(tl), pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE == "cuda")
    best, hist = -1.0, []
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); losses = []
        for x, y in tl:
            x, y = gpu_augment(x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True))
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                logits = model(x)
                loss = soft_dice_loss(logits.float(), y) + (
                    F.binary_cross_entropy_with_logits(logits.float(), y) if stage == 1 else 4 * focal_loss(logits.float(), y))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item())
        vd = val_dice(model, vl)
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "val_dice": vd})
        flag = ""
        if vd > best:
            best, flag = vd, " *best*"
            torch.save((model.module if hasattr(model, "module") else model).state_dict(), WORK / f"stage{stage}.pt")
        print(f"stage {stage} epoch {ep:2d} loss {np.mean(losses):.4f} val Dice {vd:.4f} ({time.time() - t0:.0f}s){flag}", flush=True)
    m = make_model(); m.load_state_dict(torch.load(WORK / f"stage{stage}.pt", map_location="cpu"))
    del tr, va, tl, vl; gc.collect()
    return m.to(DEVICE).eval(), best, hist

model1, best1, hist1 = train_stage(1, EPOCHS_S1)
model2, best2, hist2 = train_stage(2, EPOCHS_S2)
(WORK / "training_history.json").write_text(json.dumps({"stage1": hist1, "stage2": hist2,
                                                        "best_val_dice": {"stage1": best1, "stage2": best2}}, indent=1))

# %% [markdown]
# ## 5. Cascade inference on whole volumes (same logic as the app)

# %%
@torch.no_grad()
def predict_volume(img):
    """img: (Z, 512, 512) uint8 windowed. Returns liver (Z,512,512) bool, tumour (Z,512,512) bool, tumour prob."""
    Z = len(img); liver_p = np.zeros((Z, 512, 512), np.float32)
    for s in range(0, Z, 16):
        xs = [np.stack([cv2.resize(ch, (S1_SIZE, S1_SIZE), interpolation=cv2.INTER_AREA) for ch in stack25(img, k)])
              for k in range(s, min(Z, s + 16))]
        x = torch.from_numpy(np.stack(xs)).float().div(255).to(DEVICE)
        with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
            p = torch.sigmoid(model1(x)).float()
        p = F.interpolate(p, size=(512, 512), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
        liver_p[s:s + len(xs)] = p
    liver = liver_p > 0.5
    lab, n = ndimage.label(liver)                          # keep the largest 3-D component
    if n > 1:
        sizes = ndimage.sum(liver, lab, range(1, n + 1)); liver = lab == (1 + int(np.argmax(sizes)))
    tum_p = np.zeros((Z, 512, 512), np.float32)
    for k in range(Z):
        box = liver_box(liver[k])
        if box is None:
            continue
        x = torch.from_numpy(crop_resize(stack25(img, k), box, S2_SIZE, cv2.INTER_LINEAR)[None]).float().div(255).to(DEVICE)
        with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
            p = torch.sigmoid(model2(x)).float()[0, 0].cpu().numpy()
        y0, x0, s = box
        tum_p[k, y0:y0 + s, x0:x0 + s] = cv2.resize(p, (s, s), interpolation=cv2.INTER_LINEAR)
    tumour = (tum_p > 0.5) & liver
    return liver, tumour, tum_p

# %% [markdown]
# ## 6. Evaluation on held-out test patients
# * Liver and tumour: per-volume Dice and IoU, plus global Dice (all test voxels pooled, as in LiTS).
# * Lesions: 3-D connected components of the ground truth; a lesion counts as detected if the prediction
#   overlaps it. Recall is also given for lesions of at least 10 mm equivalent diameter.

# %%
def dice(a, b):
    s = a.sum() + b.sum(); return float(2 * (a & b).sum() / s) if s else 1.0

def iou(a, b):
    u = (a | b).sum(); return float((a & b).sum() / u) if u else 1.0

rows, pooled = [], {"li": 0, "lp": 0, "lg": 0, "ti": 0, "tp": 0, "tg": 0}
les = {"gt": 0, "det": 0, "gt10": 0, "det10": 0, "pred": 0, "pred_tp": 0}
for vid in split["test"]:
    img, seg, sp = load(vid)
    liver, tumour, _ = predict_volume(img)
    gl, gt = seg > 0, seg == 2
    r = {"volume": vid, "liver_dice": dice(liver, gl), "liver_iou": iou(liver, gl), "has_tumour": bool(gt.any())}
    if gt.any():
        r["tumour_dice"], r["tumour_iou"] = dice(tumour, gt), iou(tumour, gt)
    pooled["li"] += int((liver & gl).sum()); pooled["lp"] += int(liver.sum()); pooled["lg"] += int(gl.sum())
    pooled["ti"] += int((tumour & gt).sum()); pooled["tp"] += int(tumour.sum()); pooled["tg"] += int(gt.sum())
    voxel_ml = float(np.prod(sp)) / 1000.0
    glab, gn = ndimage.label(gt); plab, pn = ndimage.label(tumour)
    for i in range(1, gn + 1):
        comp = glab == i; vol_mm3 = comp.sum() * voxel_ml * 1000
        d_eq = (6 * vol_mm3 / math.pi) ** (1 / 3); hit = bool((comp & tumour).any())
        les["gt"] += 1; les["det"] += hit
        if d_eq >= 10: les["gt10"] += 1; les["det10"] += hit
    for i in range(1, pn + 1):
        les["pred"] += 1; les["pred_tp"] += bool(((plab == i) & gt).any())
    rows.append(r)
    print(r, flush=True)

def ms(key, sub=None):
    v = np.array([r[key] for r in (sub or rows) if key in r]); return {"mean": float(v.mean()), "std": float(v.std()), "n": int(len(v))}

metrics = {
    "test_volumes": split["test"],
    "liver_dice_per_volume": ms("liver_dice"), "liver_iou_per_volume": ms("liver_iou"),
    "liver_dice_global": 2 * pooled["li"] / max(pooled["lp"] + pooled["lg"], 1),
    "tumour_dice_per_volume": ms("tumour_dice"), "tumour_iou_per_volume": ms("tumour_iou"),
    "tumour_dice_global": 2 * pooled["ti"] / max(pooled["tp"] + pooled["tg"], 1),
    "tumour_iou_global": pooled["ti"] / max(pooled["tp"] + pooled["tg"] - pooled["ti"], 1),
    "lesion_recall": les["det"] / max(les["gt"], 1), "lesion_recall_ge10mm": les["det10"] / max(les["gt10"], 1),
    "lesion_precision": les["pred_tp"] / max(les["pred"], 1), "lesion_counts": les,
    "per_volume": rows,
}
(WORK / "metrics_test.json").write_text(json.dumps(metrics, indent=1))
print(json.dumps({k: v for k, v in metrics.items() if k != "per_volume"}, indent=1))

# %% [markdown]
# ## 7. Qualitative examples from test patients

# %%
def overlay(gray, liver, tumour, gt_t=None):
    rgb = np.stack([gray] * 3, -1).astype(np.float32)
    rgb[liver] = 0.65 * rgb[liver] + 0.35 * np.array([20, 184, 166]); rgb[tumour] = 0.4 * rgb[tumour] + 0.6 * np.array([239, 68, 68])
    rgb = rgb.astype(np.uint8)
    if gt_t is not None:
        c, _ = cv2.findContours(gt_t.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, c, -1, (250, 204, 21), 1)
    return rgb

ex_dir = WORK / "examples"; ex_dir.mkdir(exist_ok=True)
for vid in split["test"][:6]:
    img, seg, _ = load(vid)
    liver, tumour, _ = predict_volume(img)
    k = int(np.argmax((seg == 2).sum((1, 2)))) if (seg == 2).any() else int(np.argmax((seg > 0).sum((1, 2))))
    panel = np.concatenate([np.stack([img[k]] * 3, -1), overlay(img[k], liver[k], tumour[k], seg[k] == 2)], 1)
    cv2.imwrite(str(ex_dir / f"test_v{vid:03d}_s{k:03d}.png"), panel[..., ::-1])
print("examples saved (left: input; right: liver teal, predicted tumour red, expert tumour outline yellow)")

# %% [markdown]
# ## 8. Export ONNX + metadata for the LiverLenz app

# %%
class Prob(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return torch.sigmoid(self.m(x))

for name, m, size in [("liver_stage1", model1, S1_SIZE), ("tumour_stage2", model2, S2_SIZE)]:
    m = m.float().cpu().eval()
    kw = dict(input_names=["input"], output_names=["prob"], opset_version=17, dynamic_axes={"input": {0: "N"}, "prob": {0: "N"}})
    try:
        torch.onnx.export(Prob(m), torch.rand(1, 3, size, size), str(WORK / f"{name}.onnx"), dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo argument
        torch.onnx.export(Prob(m), torch.rand(1, 3, size, size), str(WORK / f"{name}.onnx"), **kw)
    print(name, f"{(WORK / f'{name}.onnx').stat().st_size / 1e6:.1f} MB")

meta = {"version": "2.0", "dataset": "LiTS (Bilic et al. 2023), Kaggle andrewmvd/liver-tumor-segmentation (+ part 2)",
        "encoder": ENCODER, "hu_window": [HU_MIN, HU_MAX], "stage1_size": S1_SIZE, "stage2_size": S2_SIZE,
        "crop_margin_px_at_512": CROP_MARGIN, "input": "NCHW float32 in [0,1]; channels = slices k-1, k, k+1 (2.5-D)",
        "orientation": "radiological display (patient right on image left, anterior at top)",
        "threshold": 0.5, "split_seed": SEED, "n_train": len(split["train"]), "n_val": len(split["val"]),
        "n_test": len(split["test"])}
(WORK / "model_meta.json").write_text(json.dumps(meta, indent=1))
print("done:", sorted(p.name for p in WORK.iterdir()))
