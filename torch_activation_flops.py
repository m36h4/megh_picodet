"""
Activation Cost Profiler -- "how torch actually runs it"
==========================================================

Instead of guessing an "ops per element" number for Hardswish / Hardsigmoid,
this script builds each variant out of *real* torch ops and profiles the
*actual* forward pass with torch's own profiler. That tells you exactly:

  - which aten kernels PyTorch dispatches for each variant (fused single-op
    vs. decomposed add/clamp/mul/div), straight from the profiler trace --
    not from a table someone typed in
  - how many times each kernel is called
  - measured wall-clock cost per kernel and per variant, on your actual
    hardware/backend

Variants covered (as requested):
    1. Hardswish -- fused          torch.nn.functional.hardswish(x)
    2. Hardswish -- split          x + 3 -> relu6/clamp -> * x -> / 6
                                    (the literal documented formula,
                                     each step a separate tensor op)
    3. Swish     -- replacement    torch.nn.functional.silu(x)   [x*sigmoid(x)]
    4. Hardsigmoid -- fused        torch.nn.functional.hardsigmoid(x)
    5. Hardsigmoid -- split        x + 3 -> relu6/clamp -> / 6
    6. Sigmoid     -- replacement  torch.sigmoid(x)

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    pip install torch

    python torch_activation_flops.py --shape 1,96,40,40 --iters 200
    python torch_activation_flops.py --shape 1,96,40,40 --device cuda --iters 500

--------------------------------------------------------------------------
WHAT YOU GET
--------------------------------------------------------------------------
  - Per-variant op dispatch table: every aten kernel actually called, how
    many times, and measured self time -- this is ground truth from
    PyTorch's profiler, not an assumption.
  - A summary comparing kernel-launch count and measured latency across
    all 6 variants, at the tensor size you gave it.
  - A "naive analytic ops" cross-check: (# elementwise kernels dispatched)
    x (numel), i.e. what a simple FLOP-style counter would report if it
    just counted kernel launches -- shown next to the *measured* time so
    you can see where the two agree/disagree (fusion helps wall-clock a
    lot more than it helps a naive op count, because it also cuts memory
    round-trips that a op-counter doesn't see at all).
"""

import argparse
import sys

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function


# --------------------------------------------------------------------------
# The variants -- built from real torch ops, not simulated
# --------------------------------------------------------------------------
def hardswish_fused(x):
    return F.hardswish(x)


def hardswish_split(x):
    # literal decomposition of x * relu6(x+3) / 6
    y = x + 3.0          # add
    y = F.relu6(y)       # relu6 (dispatches as aten::hardtanh under the hood)
    y = x * y            # mul
    y = y / 6.0          # div (elementwise scale)
    return y


def swish_replacement(x):
    return F.silu(x)     # SiLU == Swish == x * sigmoid(x), torch's fused kernel


def hardsigmoid_fused(x):
    return F.hardsigmoid(x)


def hardsigmoid_split(x):
    y = x + 3.0
    y = F.relu6(y)
    y = y / 6.0
    return y


def sigmoid_replacement(x):
    return torch.sigmoid(x)


VARIANTS = [
    ("Hardswish -- fused",        hardswish_fused),
    ("Hardswish -- split (add,clamp,mul,div)", hardswish_split),
    ("Swish (replacement)",       swish_replacement),
    ("Hardsigmoid -- fused",      hardsigmoid_fused),
    ("Hardsigmoid -- split (add,clamp,div)",   hardsigmoid_split),
    ("Sigmoid (replacement)",     sigmoid_replacement),
]


def parse_shape(s):
    return tuple(int(v) for v in s.split(","))


def profile_variant(name, fn, x, iters, device):
    # warm up (esp. important on CUDA -- first call pays kernel-compile /
    # cudnn-autotune cost that has nothing to do with the op itself)
    for _ in range(10):
        fn(x)
    if device == "cuda":
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(iters):
            with record_function(name):
                fn(x)
        if device == "cuda":
            torch.cuda.synchronize()

    key_avgs = prof.key_averages()
    # keep only actual aten kernel rows, drop our own record_function marker
    rows = [r for r in key_avgs if r.key.startswith("aten::")]
    rows.sort(key=lambda r: r.self_cpu_time_total, reverse=True)
    return rows


def fmt_us(v):
    return f"{v:,.1f} us"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", default="1,96,40,40", help="Tensor shape, comma-separated, e.g. 1,96,40,40")
    ap.add_argument("--iters", type=int, default=200, help="Forward passes to profile per variant")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available -- falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    shape = parse_shape(args.shape)
    numel = 1
    for d in shape:
        numel *= d

    torch.manual_seed(0)
    x = torch.randn(*shape, device=args.device)

    print(f"\nTensor shape: {shape}  ({numel:,} elements)  device={args.device}  iters/variant={args.iters}\n")
    print("=" * 78)

    summary = []

    for name, fn in VARIANTS:
        rows = profile_variant(name, fn, x, args.iters, args.device)

        total_calls = sum(r.count for r in rows)
        total_self_cpu = sum(r.self_cpu_time_total for r in rows)
        # per-forward-pass figures (profiler aggregates across all iters)
        calls_per_fwd = total_calls / args.iters
        us_per_fwd = total_self_cpu / args.iters

        print(f"\n{name}")
        print("-" * len(name))
        print(f"{'aten op':32} {'calls/fwd':>10} {'avg self CPU/call':>20} {'self CPU/fwd':>16}")
        for r in rows:
            calls_fwd = r.count / args.iters
            us_call = r.self_cpu_time_total / r.count if r.count else 0
            us_fwd = r.self_cpu_time_total / args.iters
            print(f"{r.key:32} {calls_fwd:>10.2f} {fmt_us(us_call):>20} {fmt_us(us_fwd):>16}")

        naive_ops = calls_per_fwd * numel  # kernel-launch-count x tensor size
        print(f"{'TOTAL':32} {calls_per_fwd:>10.2f} {'':>20} {fmt_us(us_per_fwd):>16}")
        print(f"  naive analytic 'ops' (kernel launches x numel): {naive_ops:,.0f}")

        summary.append({
            "name": name,
            "kernels": calls_per_fwd,
            "us_per_fwd": us_per_fwd,
            "naive_ops": naive_ops,
        })

    print("\n" + "=" * 78)
    print("SUMMARY (per forward pass, this tensor size)")
    print("=" * 78)
    header = f"{'Variant':42} {'kernel launches':>16} {'measured time':>16} {'naive ops':>14}"
    print(header)
    for s in summary:
        print(f"{s['name']:42} {s['kernels']:>16.2f} {fmt_us(s['us_per_fwd']):>16} {s['naive_ops']:>14,.0f}")

    print(
        "\nRead this as: 'kernel launches' and 'measured time' are ground truth "
        "from PyTorch's own profiler -- they show fusion's real benefit (fewer "
        "kernel launches -> less memory traffic -> lower wall time). 'naive "
        "ops' is what you'd get from a simple FLOP-style counter that just "
        "multiplies kernel-launch-count by tensor size; it correlates with "
        "measured time but understates how much fusion actually saves, "
        "because it can't see the memory round-trips between un-fused ops."
    )


if __name__ == "__main__":
    main()
