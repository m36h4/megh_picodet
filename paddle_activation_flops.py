"""
Activation Cost Profiler -- Paddle version ("how paddle actually runs it")
============================================================================

Same idea as torch_activation_flops.py, but built from real PaddlePaddle
ops and paddle's own profiler, using paddle's *actual* documented formulas
(which are not always identical in parameterization to torch's, even
though the functions compute the same thing):

  - paddle.nn.functional.hardswish(x)   = x * relu6(x + 3) / 6      [same
    formula/constants as torch's hardswish]
  - paddle.nn.functional.hardsigmoid(x) defaults to slope=0.1666667
    (=1/6), offset=0.5, i.e. clip(slope*x + offset, 0, 1). This is
    mathematically identical to relu6(x+3)/6, but Paddle computes it as
    one affine (mul+add) then a clip -- a *different, cheaper* primitive
    decomposition than the add->relu6->div chain used for hardswish, and
    different from how you might naively assume it decomposes if you just
    copied the hardswish pattern. This script uses Paddle's real formula,
    not torch's.
  - paddle.nn.functional.silu(x)        = Swish/SiLU, x * sigmoid(x)
  - paddle.nn.functional.sigmoid(x)     = 1 / (1 + exp(-x))

Variants profiled:
    1. Hardswish   -- fused     F.hardswish(x)
    2. Hardswish   -- split     x+3 -> relu6 -> * x -> / 6
    3. Swish (replacement)      F.silu(x)
    4. Hardsigmoid -- fused     F.hardsigmoid(x)
    5. Hardsigmoid -- split     slope*x + offset -> clip(0, 1)
                                 (Paddle's real decomposition -- 3 ops,
                                  not 4, because the /6 and +3 are folded
                                  into one affine transform up front)
    6. Sigmoid (replacement)    F.sigmoid(x)

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python paddle_activation_flops.py --shape 1,96,40,40 --iters 200
    python paddle_activation_flops.py --shape 1,96,40,40 --device gpu --iters 500

--------------------------------------------------------------------------
WHAT YOU GET
--------------------------------------------------------------------------
  - Paddle's own profiler summary (paddle.profiler.Profiler), printed in
    its native format -- this is paddle's own ground truth for which ops
    ran and how long they took. Kept as printed by paddle itself rather
    than re-parsed, since the internal profiler event schema has changed
    across paddle versions and re-parsing it risks silently misreporting.
  - A version-independent wall-clock cross-check we compute ourselves:
    plain perf_counter() timing over many warmed-up iterations, for every
    variant, side by side -- this doesn't depend on any paddle profiler
    internals and will work the same way across paddle releases.
"""

import argparse
import time

import paddle
import paddle.nn.functional as F


# --------------------------------------------------------------------------
# The variants -- built from real paddle ops, using paddle's real constants
# --------------------------------------------------------------------------
def hardswish_fused(x):
    return F.hardswish(x)


def hardswish_split(x):
    # x * relu6(x + 3) / 6 -- same formula/constants paddle's fused op uses
    y = x + 3.0
    y = F.relu6(y)
    y = x * y
    y = y / 6.0
    return y


def swish_replacement(x):
    return F.silu(x)


def hardsigmoid_fused(x):
    return F.hardsigmoid(x)  # paddle defaults: slope=0.1666667, offset=0.5


def hardsigmoid_split(x):
    # paddle's real decomposition: clip(slope*x + offset, 0, 1)
    slope = 0.1666667
    offset = 0.5
    y = x * slope
    y = y + offset
    y = paddle.clip(y, min=0.0, max=1.0)
    return y


def sigmoid_replacement(x):
    return F.sigmoid(x)


VARIANTS = [
    ("Hardswish -- fused",                       hardswish_fused),
    ("Hardswish -- split (add,relu6,mul,div)",   hardswish_split),
    ("Swish (replacement)",                      swish_replacement),
    ("Hardsigmoid -- fused",                     hardsigmoid_fused),
    ("Hardsigmoid -- split (mul,add,clip)",      hardsigmoid_split),
    ("Sigmoid (replacement)",                    sigmoid_replacement),
]


def parse_shape(s):
    return tuple(int(v) for v in s.split(","))


def sync(device):
    if device == "gpu":
        paddle.device.cuda.synchronize()


def wall_clock_time(fn, x, iters, device):
    for _ in range(10):  # warm-up
        fn(x)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    sync(device)
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e6  # microseconds per forward pass


def run_paddle_profiler(name, fn, x, iters, device):
    try:
        from paddle.profiler import Profiler, ProfilerTarget, SortedKeys
    except Exception as e:
        print(f"  (paddle.profiler unavailable in this paddle build: {e})")
        return

    targets = [ProfilerTarget.CPU]
    if device == "gpu":
        targets.append(ProfilerTarget.GPU)

    warmup = max(1, iters // 10)
    prof = Profiler(targets=targets, scheduler=(warmup, iters))
    prof.start()
    for _ in range(iters):
        fn(x)
        prof.step()
    prof.stop()

    print(f"\n  --- paddle.profiler summary: {name} ---")
    try:
        prof.summary(op_detail=True, sorted_by=SortedKeys.CPUTotal, thread_sep=False)
    except TypeError:
        # older/newer paddle releases vary in accepted kwargs -- fall back
        # to the plain call rather than guess-fitting our way to a match
        prof.summary()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", default="1,96,40,40", help="Tensor shape, comma-separated, e.g. 1,96,40,40")
    ap.add_argument("--iters", type=int, default=200, help="Forward passes per variant")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--skip-paddle-profiler", action="store_true",
                     help="Only run the wall-clock cross-check, skip paddle.profiler.Profiler "
                          "(useful if your paddle build's profiler output is noisy/unavailable)")
    args = ap.parse_args()

    paddle.set_device(args.device)

    shape = parse_shape(args.shape)
    numel = 1
    for d in shape:
        numel *= d

    paddle.seed(0)
    x = paddle.randn(shape)

    print(f"\nTensor shape: {shape}  ({numel:,} elements)  device={args.device}  iters/variant={args.iters}\n")
    print("=" * 78)

    wall_times = []
    for name, fn in VARIANTS:
        print(f"\n{name}")
        print("-" * len(name))

        if not args.skip_paddle_profiler:
            run_paddle_profiler(name, fn, x, args.iters, args.device)

        us = wall_clock_time(fn, x, args.iters, args.device)
        print(f"  wall-clock: {us:,.1f} us/forward pass (perf_counter, {args.iters} iters, warmed up)")
        wall_times.append((name, us))

    print("\n" + "=" * 78)
    print("SUMMARY -- wall-clock cross-check (per forward pass, this tensor size)")
    print("=" * 78)
    print(f"{'Variant':42} {'us/forward':>14}")
    for name, us in wall_times:
        print(f"{name:42} {us:>14,.1f}")

    print(
        "\nThe paddle.profiler sections above (if shown) are Paddle's own "
        "op-level ground truth for this run -- which kernels actually got "
        "dispatched and how long each took, straight from Paddle itself. "
        "The wall-clock summary is a simple, version-independent cross-check: "
        "fused ops should consistently come out faster than their split "
        "counterparts because they avoid the extra memory round-trips "
        "between separate kernel launches, even though the split version "
        "computes the exact same math."
    )


if __name__ == "__main__":
    main()
