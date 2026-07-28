"""
Layerwise (per-section) params / MACs / FLOPs profiler.

Reproduces the "Backbone / FPN / Head / Aux" breakdown style from the
nanodet-plus spreadsheet. Convention: FLOPs = 2 x MACs.

Run this file directly:  python layerwise_profile.py
It will run a small self-contained DEMO model first (no torch/paddle
setup required beyond `pip install torch fvcore`), so you see real
output immediately. Then swap in your real model as shown at the bottom.
"""

# =====================================================================
# PART A — PyTorch (use for NanoDet-Plus)
# =====================================================================
# pip install torch fvcore

def profile_pytorch_model(model, input_tensor, section_map, model_name="model"):
    """
    model:        an nn.Module, will be set to eval() internally
    input_tensor: one dummy input, e.g. torch.randn(1, 3, 320, 320)
    section_map:  dict mapping a TOP-LEVEL child module name -> section label
                  e.g. {"backbone": "Backbone", "fpn": "FPN",
                        "head": "Head", "aux_head": "Aux (training-only)"}
                  Any top-level child not listed lands in "Other".
    """
    import torch
    from fvcore.nn import FlopCountAnalysis

    model.eval()
    with torch.no_grad():
        fca = FlopCountAnalysis(model, input_tensor)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        macs_by_module = fca.by_module()   # cumulative per-module MACs
        total_macs = fca.total()

    # IMPORTANT: by_module() values are CUMULATIVE (a module's count already
    # includes all its children). So for section totals we take ONLY the
    # direct top-level key's value -- never sum nested keys too, or you'll
    # multiply-count the same MACs several times over.
    section_macs, section_params = {}, {}
    for child_name, child_module in model.named_children():
        label = section_map.get(child_name, "Other")
        section_macs[label] = section_macs.get(label, 0) + macs_by_module.get(child_name, 0)
        section_params[label] = section_params.get(label, 0) + sum(
            p.numel() for p in child_module.parameters()
        )

    total_params = sum(p.numel() for p in model.parameters())

    _print_report(model_name, total_params, total_macs, section_params, section_macs)


# =====================================================================
# PART B — PaddlePaddle (use for PicoDet)
# =====================================================================
# pip install paddlepaddle

def profile_paddle_model(model, input_size, section_map, model_name="model"):
    """
    Same idea as Part A, but for Paddle. Uses forward hooks on each
    top-level submodule to capture MACs via paddle's own built-in
    paddle.flops() utility, run on that submodule alone with the real
    input shape it received.

    model:        a paddle.nn.Layer
    input_size:   e.g. (1, 3, 320, 320)
    section_map:  dict mapping a TOP-LEVEL sublayer name -> section label,
                  same shape as Part A's section_map.
    """
    import paddle

    section_macs, section_params, hooks = {}, {}, []

    def make_hook(label):
        def hook(layer, inputs, output):
            in_shape = list(inputs[0].shape) if isinstance(inputs, (list, tuple)) else list(inputs.shape)
            try:
                flops = paddle.flops(layer, in_shape, print_detail=False)
                macs = flops // 2
            except Exception as e:
                print(f"  [warn] could not profile a layer in section '{label}': {e}")
                macs = 0
            section_macs[label] = section_macs.get(label, 0) + macs
        return hook

    for child_name, child_layer in model.named_children():
        label = section_map.get(child_name, "Other")
        hooks.append(child_layer.register_forward_post_hook(make_hook(label)))
        section_params[label] = section_params.get(label, 0) + sum(
            p.numel() for p in child_layer.parameters()
        )

    model.eval()
    dummy = paddle.randn(input_size)
    with paddle.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    total_params = sum(p.numel() for p in model.parameters())
    total_macs = sum(section_macs.values())

    _print_report(model_name, total_params, total_macs, section_params, section_macs)

    print("\nNote: this sums MACs from per-section forward hooks, which is an\n"
          "approximation (submodules with non-standard/custom ops may be\n"
          "misreported by paddle.flops -- cross-check the total against:\n"
          "  paddle.flops(model, input_size, print_detail=True)\n"
          "which prints Paddle's own full per-layer table for the whole model.")


# =====================================================================
# Shared report printer
# =====================================================================

def _print_report(model_name, total_params, total_macs, section_params, section_macs):
    print(f"\n{model_name} - Model Summary")
    print(f"{'Total Params':30s}{total_params:>15,}")
    print(f"{'Total MACs':30s}{total_macs:>15,}")
    print(f"{'Total FLOPs':30s}{2*total_macs:>15,}")
    print("\nBreakdown by section")
    print(f"{'Section':<24s}{'Params':>15s}{'MACs':>15s}{'FLOPs':>15s}{'% of Params':>14s}")
    all_sections = sorted(set(section_params) | set(section_macs))
    for sec in all_sections:
        p = section_params.get(sec, 0)
        m = section_macs.get(sec, 0)
        pct = 100.0 * p / total_params if total_params else 0.0
        print(f"{sec:<24s}{p:>15,}{m:>15,}{2*m:>15,}{pct:>13.2f}%")


# =====================================================================
# DEMO — proves the script actually produces output, no real model needed
# =====================================================================

def _demo():
    import torch
    import torch.nn as nn

    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        def forward(self, x):
            return self.conv(x)

    class DummyFPN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(32, 32, 1)
        def forward(self, x):
            return self.conv(x)

    class DummyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(32, 5, 3, padding=1)
        def forward(self, x):
            return self.conv(x)

    class DummyAuxHead(nn.Module):
        # simulates NanoDet-Plus's AGM: exists, but NOT called in forward()
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(32, 5, 3, padding=1)
        def forward(self, x):
            return self.conv(x)

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = DummyBackbone()
            self.fpn = DummyFPN()
            self.head = DummyHead()
            self.aux_head = DummyAuxHead()  # present, but skipped below
        def forward(self, x):
            x = self.backbone(x)
            x = self.fpn(x)
            x = self.head(x)   # aux_head deliberately not called -> 0 MACs
            return x

    model = DummyModel()
    dummy_input = torch.randn(1, 3, 320, 320)

    profile_pytorch_model(
        model,
        dummy_input,
        section_map={
            "backbone": "Backbone",
            "fpn": "FPN",
            "head": "Head",
            "aux_head": "Aux (training-only)",
        },
        model_name="demo-model (proves the script works)",
    )


if __name__ == "__main__":
    _demo()

    # ---- Once the demo above prints correctly, replace it with your real
    # ---- model. Example for NanoDet-Plus:
    #
    # import torch
    # from nanodet.model.arch import build_model
    # from nanodet.util import cfg, load_config
    # load_config(cfg, "config/nanodet-plus-m_320.yml")
    # model = build_model(cfg.model)
    # dummy_input = torch.randn(1, 3, 320, 320)
    # profile_pytorch_model(
    #     model, dummy_input,
    #     section_map={"backbone": "Backbone", "fpn": "FPN", "head": "Head",
    #                  "aux_fpn": "Aux (training-only)",
    #                  "aux_head": "Aux (training-only)"},
    #     model_name="nanodet-plus",
    # )
    #
    # Example for PicoDet (Paddle):
    #
    # from ppdet.core.workspace import load_config, create
    # cfg = load_config("configs/picodet/picodet_s_320_coco.yml")
    # model = create(cfg.architecture)
    # profile_paddle_model(
    #     model, input_size=(1, 3, 320, 320),
    #     section_map={"backbone": "Backbone", "neck": "FPN", "head": "Head"},
    #     model_name="picodet-s",
    # )
