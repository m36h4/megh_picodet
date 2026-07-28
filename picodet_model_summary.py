"""
PicoDet (PaddleDetection) Model Summary & FLOPs / Params Calculator
=====================================================================

Produces the same kind of report as the nanodet-plus "Model Summary" sheet:
  - Overall totals (params / MACs / FLOPs)
  - Breakdown by section (Backbone / Neck(FPN) / Head)
  - Full per-layer detail (every leaf layer, its type, shape, params, MACs, FLOPs)

It works directly against a real PaddleDetection PicoDet model (built from a
config + optional weights), by hooking every leaf `paddle.nn.Layer` during a
single dummy forward pass and computing MACs analytically from the captured
input/output tensor shapes -- no third-party flops library required.

--------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------
    pip install paddlepaddle          # or paddlepaddle-gpu
    pip install pandas openpyxl
    # PaddleDetection checked out locally, with its root on PYTHONPATH
    # (this is what provides the `ppdet` package):
    #   git clone https://github.com/PaddlePaddle/PaddleDetection.git

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    cd PaddleDetection   # so `ppdet` is importable, OR pass --ppdet-root

    python picodet_model_summary.py \
        --config configs/picodet/picodet_s_320_coco_lcnet.yml \
        --weights output/picodet_s_320_coco_lcnet/model_final.pdparams \
        --input-size 320 \
        --out picodet_s_320_summary.xlsx

    # weights are optional -- params/MACs/FLOPs don't depend on values,
    # only on architecture + input size, so you can omit --weights and
    # still get an exact structural report.

--------------------------------------------------------------------------
NOTES / ASSUMPTIONS (edit SECTION_MAP below if your config differs)
--------------------------------------------------------------------------
    - PicoDet architectures in PaddleDetection expose top-level attributes
      `backbone`, `neck` (CSPPAN), and `head` (PicoHead / PicoHeadV2 /
      PicoFeatureFusionHead). Every leaf layer's full dotted name is
      bucketed into a section by matching its first path component against
      SECTION_MAP. Anything unmatched goes into "Other".
    - FLOPs = 2 x MACs for Conv2D / Linear (standard multiply-add
      convention), matching the ratio in the reference nanodet-plus sheet.
    - BatchNorm / activation / pooling layers contribute a small
      elementwise MACs/FLOPs term; depthwise-separable convs are handled
      correctly via the layer's `groups` attribute.
    - This script profiles inference-time compute only (a single forward
      pass in eval mode). PicoDet (unlike nanodet-plus) has no
      training-only auxiliary branch, so there is no "Aux" row here.
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Section bucketing -- edit if your PicoDet variant names things differently
# --------------------------------------------------------------------------
SECTION_MAP = [
    ("backbone", "Backbone"),
    ("neck", "Neck (CSP-PAN)"),
    ("fpn", "Neck (CSP-PAN)"),
    ("head", "Head"),
]


def bucket_section(full_name: str) -> str:
    first = full_name.split(".")[0].lower()
    for key, label in SECTION_MAP:
        if key in first:
            return label
    return "Other"


# --------------------------------------------------------------------------
# Per-layer-type MACs estimators (filled in once paddle is imported)
# --------------------------------------------------------------------------
def build_mac_functions(paddle):
    nn = paddle.nn

    def conv_macs(layer, in_shape, out_shape):
        # in_shape/out_shape: (N, C, H, W)
        _, out_c, out_h, out_w = out_shape
        in_c = layer._in_channels
        k_h, k_w = layer._kernel_size
        groups = getattr(layer, "_groups", 1) or 1
        return (in_c // groups) * out_c * k_h * k_w * out_h * out_w

    def linear_macs(layer, in_shape, out_shape):
        batch = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
        return layer.weight.shape[0] * layer.weight.shape[1] * batch

    def elementwise_macs(layer, in_shape, out_shape):
        return int(np.prod(out_shape))

    def zero_macs(layer, in_shape, out_shape):
        return 0

    table = {}
    for cls in (nn.Conv2D, nn.Conv2DTranspose):
        table[cls] = conv_macs
    for cls in (nn.Linear,):
        table[cls] = linear_macs
    for cls in (nn.BatchNorm2D, nn.BatchNorm, nn.SyncBatchNorm, nn.GroupNorm):
        table[cls] = elementwise_macs
    for cls in (
        nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.Hardswish, nn.Hardsigmoid,
        nn.Sigmoid, nn.Swish, nn.AvgPool2D, nn.MaxPool2D,
        nn.AdaptiveAvgPool2D, nn.Upsample, nn.Dropout, nn.Identity,
    ):
        table[cls] = elementwise_macs
    return table


# --------------------------------------------------------------------------
# Core profiler
# --------------------------------------------------------------------------
def profile_model(paddle, model, input_size, extra_inputs=None):
    """
    Runs one forward pass, hooking every leaf sublayer.
    Returns a list of dict rows: name, type, params, macs, flops, out_shape
    """
    mac_funcs = build_mac_functions(paddle)
    rows = []
    hooks = []

    def is_leaf(layer):
        return len(list(layer.children())) == 0

    def make_hook(name, layer):
        def hook(lyr, inputs, output):
            in_shape = tuple(inputs[0].shape) if inputs else ()
            out_shape = tuple(output.shape) if hasattr(output, "shape") else ()
            fn = mac_funcs.get(type(lyr))
            if fn is not None and in_shape and out_shape:
                try:
                    macs = int(fn(lyr, in_shape, out_shape))
                except Exception:
                    macs = 0
            else:
                macs = 0
            params = sum(p.numel().item() if hasattr(p.numel(), "item") else int(p.numel())
                         for p in lyr.parameters(include_sublayers=False))
            rows.append({
                "name": name,
                "type": type(lyr).__name__,
                "in_shape": in_shape,
                "out_shape": out_shape,
                "params": int(params),
                "macs": macs,
                "flops": macs * 2,
            })
        return hook

    for name, layer in model.named_sublayers():
        if is_leaf(layer):
            hooks.append(layer.register_forward_post_hook(make_hook(name, layer)))

    model.eval()
    dummy_image = paddle.randn([1, 3, input_size, input_size], dtype="float32")
    inputs = {"image": dummy_image}
    if extra_inputs:
        inputs.update(extra_inputs)
    else:
        inputs["im_shape"] = paddle.to_tensor([[input_size, input_size]], dtype="float32")
        inputs["scale_factor"] = paddle.to_tensor([[1.0, 1.0]], dtype="float32")

    with paddle.no_grad():
        try:
            model(inputs)
        except Exception as e:
            for h in hooks:
                h.remove()
            raise RuntimeError(
                f"Forward pass failed with inputs {list(inputs.keys())}. "
                f"Your PicoDet variant may expect different input keys -- "
                f"pass --extra-inputs or edit profile_model(). Original error: {e}"
            )

    for h in hooks:
        h.remove()

    return rows


def summarize(rows):
    df = pd.DataFrame(rows)
    df["section"] = df["name"].apply(bucket_section)

    total_params = df["params"].sum()
    total_macs = df["macs"].sum()
    total_flops = df["flops"].sum()
    total_layers = len(df)

    by_section = (
        df.groupby("section")
        .agg(layers=("name", "count"), params=("params", "sum"),
             macs=("macs", "sum"), flops=("flops", "sum"))
        .reset_index()
    )
    by_section["% of total params"] = (
        by_section["params"] / total_params * 100 if total_params else 0
    )

    overall = pd.DataFrame([{
        "Total layers": total_layers,
        "Total Params": total_params,
        "Total MACs": total_macs,
        "Total FLOPs": total_flops,
    }])

    return overall, by_section.sort_values("params", ascending=False), df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="PaddleDetection PicoDet yml config path")
    ap.add_argument("--weights", default=None, help="Optional .pdparams checkpoint")
    ap.add_argument("--input-size", type=int, default=320)
    ap.add_argument("--ppdet-root", default=None,
                     help="Path to PaddleDetection repo root, if not already on PYTHONPATH")
    ap.add_argument("--out", default="picodet_summary.xlsx")
    args = ap.parse_args()

    if args.ppdet_root:
        sys.path.insert(0, args.ppdet_root)

    import paddle
    from ppdet.core.workspace import load_config, create
    from ppdet.utils.checkpoint import load_weight

    cfg = load_config(args.config)
    model = create(cfg.architecture)

    if args.weights:
        load_weight(model, args.weights)

    rows = profile_model(paddle, model, args.input_size)
    overall, by_section, detail = summarize(rows)

    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    print("\n=== Overall totals ===")
    print(overall.to_string(index=False))
    print("\n=== Breakdown by section ===")
    print(by_section.to_string(index=False))

    with pd.ExcelWriter(args.out) as writer:
        overall.to_excel(writer, sheet_name="Overall", index=False)
        by_section.to_excel(writer, sheet_name="By Section", index=False)
        detail.to_excel(writer, sheet_name="Per-Layer Detail", index=False)

    print(f"\nSaved full report to {args.out}")


if __name__ == "__main__":
    main()
