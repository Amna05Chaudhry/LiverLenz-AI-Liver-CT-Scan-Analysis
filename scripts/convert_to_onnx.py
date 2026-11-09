"""Convert the Keras .h5 segmentation model to ONNX without TensorFlow.

Reads the model config and weights straight from the HDF5 file with h5py and
builds an equivalent ONNX graph (NHWC in/out, NCHW inside). Supports the layer
types used by the notebook's FCN / SegNet models: Conv2D, BatchNormalization,
Activation, MaxPooling2D, UpSampling2D, Cropping2D, Dropout.

    python scripts/convert_to_onnx.py models/liver_segmentation_model.h5 models/liver_segmentation_model.onnx

Verify against Keras (needs TensorFlow, dev only)::

    python scripts/convert_to_onnx.py ... --verify
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def layer_weights(f: h5py.File, name: str) -> dict[str, np.ndarray]:
    g = f["model_weights"][name]
    out = {}
    for wn in g.attrs.get("weight_names", []):
        wn = wn.decode() if isinstance(wn, bytes) else wn
        out[wn.split("/")[-1].split(":")[0]] = np.array(g[wn])
    return out


def convert(h5_path: str, onnx_path: str) -> None:
    f = h5py.File(h5_path, "r")
    cfg = json.loads(f.attrs["model_config"])
    if cfg["class_name"] != "Sequential":
        raise SystemExit("Only Sequential models are supported")
    layers = cfg["config"]["layers"]

    nodes, inits = [], []
    cur = "nchw_in"
    nodes.append(helper.make_node("Transpose", ["input"], [cur], perm=[0, 3, 1, 2]))
    shape = None

    def act(x, a, tag):
        if a in (None, "linear"):
            return x
        op = {"relu": "Relu", "sigmoid": "Sigmoid", "softmax": "Softmax", "tanh": "Tanh"}[a]
        y = f"{tag}_{a}"
        nodes.append(helper.make_node(op, [x], [y]))
        return y

    for L in layers:
        cls, c = L["class_name"], L["config"]
        name = c.get("name")
        if cls == "InputLayer":
            shape = c.get("batch_input_shape") or c.get("batch_shape")
            continue
        if shape is None and "batch_input_shape" in c:
            shape = c["batch_input_shape"]
        if cls == "Conv2D":
            w = layer_weights(f, name)
            k = w["kernel"].transpose(3, 2, 0, 1)  # HWIO -> OIHW
            inits.append(numpy_helper.from_array(k.astype(np.float32), f"{name}_W"))
            ins = [cur, f"{name}_W"]
            if c.get("use_bias", True):
                inits.append(numpy_helper.from_array(w["bias"].astype(np.float32), f"{name}_B"))
                ins.append(f"{name}_B")
            assert tuple(c.get("dilation_rate", (1, 1))) == (1, 1)
            pad = "SAME_UPPER" if c["padding"] == "same" else "VALID"
            nodes.append(helper.make_node("Conv", ins, [name], kernel_shape=list(c["kernel_size"]),
                                          strides=list(c["strides"]), auto_pad=pad))
            cur = act(name, c.get("activation"), name)
        elif cls == "BatchNormalization":
            w = layer_weights(f, name)
            n = w["moving_mean"].shape[0]
            gamma = w.get("gamma", np.ones(n, np.float32))
            beta = w.get("beta", np.zeros(n, np.float32))
            for suf, arr in [("g", gamma), ("b", beta), ("m", w["moving_mean"]), ("v", w["moving_variance"])]:
                inits.append(numpy_helper.from_array(arr.astype(np.float32), f"{name}_{suf}"))
            nodes.append(helper.make_node("BatchNormalization", [cur, f"{name}_g", f"{name}_b", f"{name}_m", f"{name}_v"],
                                          [name], epsilon=float(c.get("epsilon", 1e-3))))
            cur = name
        elif cls == "Activation":
            cur = act(cur, c["activation"], name)
        elif cls == "MaxPooling2D":
            strides = c.get("strides") or c["pool_size"]
            pad = "SAME_UPPER" if c.get("padding") == "same" else "VALID"
            nodes.append(helper.make_node("MaxPool", [cur], [name], kernel_shape=list(c["pool_size"]),
                                          strides=list(strides), auto_pad=pad))
            cur = name
        elif cls == "UpSampling2D":
            sy, sx = c["size"]
            inits.append(numpy_helper.from_array(np.array([1, 1, sy, sx], np.float32), f"{name}_scales"))
            nodes.append(helper.make_node("Resize", [cur, "", f"{name}_scales"], [name], mode="nearest",
                                          coordinate_transformation_mode="asymmetric", nearest_mode="floor"))
            cur = name
        elif cls == "Cropping2D":
            (t, b), (l, r) = c["cropping"]
            if t or b or l or r:
                raise NotImplementedError("non-zero Cropping2D")
        elif cls == "Dropout":
            continue
        else:
            raise NotImplementedError(cls)

    nodes.append(helper.make_node("Transpose", [cur], ["output"], perm=[0, 2, 3, 1]))
    h, w_, ch = shape[1], shape[2], shape[3]
    graph = helper.make_graph(
        nodes, "liverlenz_segnet",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", h, w_, ch])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", h, w_, 1])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="liverlenz-h5-to-onnx")
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)
    print(f"Saved {onnx_path} ({Path(onnx_path).stat().st_size / 1e6:.1f} MB, {len(nodes)} nodes)")


def verify(h5_path: str, onnx_path: str, n: int = 8) -> None:
    import keras
    import onnxruntime as ort

    km = keras.models.load_model(h5_path, compile=False, custom_objects={"dice_coefficient": lambda a, b: 0})
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = rng.random((n, 128, 128, 3), dtype=np.float32)
    a = np.asarray(km(x, training=False))
    b = sess.run(None, {"input": x})[0]
    print(f"max |keras - onnx| = {np.abs(a - b).max():.2e}, mask agreement = {((a > .5) == (b > .5)).mean():.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5", nargs="?", default="models/liver_segmentation_model.h5")
    ap.add_argument("onnx", nargs="?", default="models/liver_segmentation_model.onnx")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    convert(a.h5, a.onnx)
    if a.verify:
        verify(a.h5, a.onnx)
