"""
Parse a PaddleDetection training log (the raw terminal/stdout text) and
plot:
  1. Training loss curves (total loss, loss_vfl, loss_bbox, loss_dfl) vs epoch
  2. Eval mAP@0.5:0.95 vs epoch (from the periodic --eval runs during training)

Usage:
    python plot_loss_curve.py --log training_log.txt --out loss_curve.png
"""

import argparse
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_log(log_path):
    epoch_loss = {}  # epoch -> list of (loss_vfl, loss_bbox, loss_dfl, loss)
    eval_ap = []      # list of (epoch_at_time_of_eval, ap)

    line_re = re.compile(
        r"Epoch:\s*\[(\d+)\].*?"
        r"loss_vfl:\s*([\d.]+)\s+"
        r"loss_bbox:\s*([\d.]+)\s+"
        r"loss_dfl:\s*([\d.]+)\s+"
        r"loss:\s*([\d.]+)"
    )
    ap_re = re.compile(
        r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=\s*all \| maxDets=100 \] = ([\d.]+)"
    )
    best_ap_re = re.compile(r"Best test bbox ap is ([\d]+\.[\d]+)")

    last_epoch_seen = 0
    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            m = line_re.search(line)
            if m:
                epoch = int(m.group(1))
                last_epoch_seen = epoch
                vfl, bbox, dfl, total = (float(m.group(i)) for i in range(2, 6))
                epoch_loss.setdefault(epoch, []).append((vfl, bbox, dfl, total))
                continue
            m2 = best_ap_re.search(line)
            if m2:
                eval_ap.append((last_epoch_seen, float(m2.group(1))))

    # average losses within each epoch (log has multiple lines per epoch)
    epochs_sorted = sorted(epoch_loss.keys())
    avg_vfl, avg_bbox, avg_dfl, avg_total = [], [], [], []
    for e in epochs_sorted:
        vals = epoch_loss[e]
        avg_vfl.append(sum(v[0] for v in vals) / len(vals))
        avg_bbox.append(sum(v[1] for v in vals) / len(vals))
        avg_dfl.append(sum(v[2] for v in vals) / len(vals))
        avg_total.append(sum(v[3] for v in vals) / len(vals))

    return epochs_sorted, avg_vfl, avg_bbox, avg_dfl, avg_total, eval_ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True, help='Path to saved training log text file')
    ap.add_argument('--out', default='loss_curve.png')
    args = ap.parse_args()

    epochs, vfl, bbox, dfl, total, eval_ap = parse_log(args.log)

    if not epochs:
        print("No training loss lines matched — check the log format/path.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss curve ---
    ax = axes[0]
    ax.plot(epochs, total, label='Total loss', linewidth=2, color='black')
    ax.plot(epochs, vfl, label='loss_vfl (classification)', alpha=0.7)
    ax.plot(epochs, bbox, label='loss_bbox (GIoU)', alpha=0.7)
    ax.plot(epochs, dfl, label='loss_dfl (distribution focal)', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss vs Epoch')
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Eval mAP curve ---
    ax2 = axes[1]
    if eval_ap:
        eval_epochs = [e for e, _ in eval_ap]
        eval_vals = [v for _, v in eval_ap]
        ax2.plot(eval_epochs, eval_vals, marker='o', color='tab:blue')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP@0.5:0.95 (val)')
        ax2.set_title('Validation mAP vs Epoch')
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No eval mAP lines found', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")

    print("\n--- Summary ---")
    print(f"Epochs parsed: {epochs[0]} to {epochs[-1]}")
    print(f"Final total loss: {total[-1]:.4f} (started at {total[0]:.4f})")
    if eval_ap:
        print(f"Eval mAP progression: {eval_ap}")


if __name__ == '__main__':
    main()
