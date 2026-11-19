"""Figures: (1) OpenCV QC overlay of GT vs prediction contours on a coronal CT slice
through the femoral head, (2) Bland-Altman plot of femoral-head diameter, (3) Dice boxplot.

Usage: python scripts/make_figures.py DATA_DIR --case s0011 [--side right] --preds preds --results results

Note: the overlay shows the patient's left on the viewer's left (not the radiological convention).
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from femur3d.io import load, load_mask  # noqa: E402
from femur3d.geometry import femoral_head  # noqa: E402


def window(ct_slice, lo=-400, hi=1500):
    """Bone window -> uint8."""
    return (np.clip((ct_slice - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def to_display(a):
    """Coronal slice arr[x, z] -> image with superior at the top."""
    return np.ascontiguousarray(np.rot90(a))


def draw(img_bgr, mask2d, colour):
    cnts, _ = cv2.findContours(mask2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_bgr, cnts, -1, colour, 2)


def qc_overlay(data_dir, case, preds, mode, out_png, side="left"):
    ct, spacing = load(data_dir / case / "ct.nii.gz", np.float32)
    gt, _ = load_mask(data_dir / case / "segmentations" / f"femur_{side}.nii.gz")
    hip, _ = load_mask(data_dir / case / "segmentations" / f"hip_{side}.nii.gz")
    pr, _ = load_mask(preds / mode / case / f"femur_{side}.nii.gz")
    head = femoral_head(gt, hip, spacing)
    y = int(round(head["centre_mm"][1] / spacing[1])) if head else gt.shape[1] // 2
    gray = to_display(window(ct[:, y, :]))
    m_gt, m_pr = to_display(gt[:, y, :]).astype(np.uint8), to_display(pr[:, y, :]).astype(np.uint8)
    # Resample to square pixels (3 px per mm) BEFORE drawing, so anatomy is not stretched
    # and contours stay crisp.
    h, w = gray.shape
    size = (int(w * spacing[0] * 3), int(h * spacing[2] * 3))
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_LINEAR)
    m_gt = cv2.resize(m_gt, size, interpolation=cv2.INTER_NEAREST)
    m_pr = cv2.resize(m_pr, size, interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    draw(img, m_gt, (0, 255, 0))   # green = ground truth
    draw(img, m_pr, (0, 0, 255))   # red = prediction
    cv2.putText(img, f"{case} {mode} | green=GT red=pred | slice y={y}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_png), img)


def bland_altman(head_csv, out_png):
    df = pd.read_csv(head_csv).dropna(subset=["gt_diam_mm", "pred_diam_mm"])
    fig, axes = plt.subplots(1, df["mode"].nunique(), figsize=(10, 4), squeeze=False)
    for ax, (mode, d) in zip(axes[0], df.groupby("mode")):
        mean = (d.gt_diam_mm + d.pred_diam_mm) / 2
        diff = d.pred_diam_mm - d.gt_diam_mm
        bias, sd = diff.mean(), diff.std()
        ax.scatter(mean, diff, s=18)
        for v, ls in [(bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")]:
            ax.axhline(v, ls=ls, c="k", lw=1)
        ax.ticklabel_format(useOffset=False)
        ax.set(title=f"{mode}: bias {bias:.2f} mm, LoA ±{1.96*sd:.2f}",
               xlabel="mean head diameter (mm)", ylabel="pred - GT (mm)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


def dice_box(seg_csv, out_png):
    df = pd.read_csv(seg_csv)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels, data = [], []
    for (bone, mode), d in df.groupby(["bone", "mode"]):
        labels.append(f"{bone}\n{mode}")
        data.append(d.dice.dropna())
    ax.boxplot(data)
    ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=7)
    ax.set_ylabel("Dice")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--case", required=True)
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--preds", type=Path, default=Path("preds"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    a = ap.parse_args()
    fig_dir = a.results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("full", "fast"):
        if (a.preds / mode / a.case).exists():
            qc_overlay(a.data_dir, a.case, a.preds, mode, fig_dir / f"qc_{a.case}_{mode}_{a.side}.png", side=a.side)
    bland_altman(a.results / "femoral_head.csv", fig_dir / "bland_altman_head_diameter.png")
    dice_box(a.results / "segmentation_metrics.csv", fig_dir / "dice_boxplot.png")
    print("figures in", fig_dir)


if __name__ == "__main__":
    main()
