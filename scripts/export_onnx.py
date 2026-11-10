"""Export the Kaggle-trained PyTorch checkpoints to ONNX for the app.

    python scripts/export_onnx.py path/to/stage1.pt path/to/stage2.pt

Writes models/liver_stage1.onnx, models/tumour_stage2.onnx and models/model_meta.json.
Same architecture and export code as the notebook (U-Net, ResNet-18 encoder, sigmoid output, NCHW).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent


class Prob(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.sigmoid(self.m(x))


def export(ckpt: Path, out: Path, size: int = 256):
    m = smp.Unet("resnet18", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    torch.onnx.export(Prob(m), torch.rand(1, 3, size, size), str(out), input_names=["input"], output_names=["prob"],
                      opset_version=17, dynamic_axes={"input": {0: "N"}, "prob": {0: "N"}}, dynamo=False)
    print(f"{out.name}: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    s1, s2 = map(Path, sys.argv[1:3])
    (ROOT / "models").mkdir(exist_ok=True)
    export(s1, ROOT / "models" / "liver_stage1.onnx")
    export(s2, ROOT / "models" / "tumour_stage2.onnx")
    meta = {"version": "2.0", "dataset": "LiTS (Bilic et al. 2023), Kaggle andrewmvd/liver-tumor-segmentation (+ part 2)",
            "encoder": "resnet18", "hu_window": [-100.0, 250.0], "stage1_size": 256, "stage2_size": 256,
            "crop_margin_px_at_512": 16, "input": "NCHW float32 in [0,1]; channels = slices k-1, k, k+1 (2.5-D)",
            "orientation": "radiological display (patient right on image left, anterior at top)", "threshold": 0.5,
            "split_seed": 42, "n_train": 101, "n_val": 10, "n_test": 20,
            "checkpoints": {"stage1": "epoch 12, val Dice 0.9286", "stage2": "epoch 16, val Dice 0.7512"},
            "trained_on": "Kaggle GPU T4 x2, notebook amna0chaudhry/notebookdeb2a5bd66 version 1"}
    (ROOT / "models" / "model_meta.json").write_text(json.dumps(meta, indent=1))
