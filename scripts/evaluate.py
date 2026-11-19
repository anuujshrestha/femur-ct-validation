"""Compare predictions with ground truth: Dice, HD95, ASSD per bone, and
femoral-head diameter / centre error per side. Also exports STL meshes.

Usage: python scripts/evaluate.py DATA_DIR cases.csv --preds preds --out results
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from femur3d.io import load_mask  # noqa: E402
from femur3d.metrics import dice, surface_metrics  # noqa: E402
from femur3d.geometry import femoral_head, mask_to_mesh  # noqa: E402

BONES = ["femur_left", "femur_right", "hip_left", "hip_right"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("cases", type=Path)
    ap.add_argument("--preds", type=Path, default=Path("preds"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--modes", nargs="+", default=["full", "fast"])
    ap.add_argument("--stl", action="store_true", help="export STL meshes for each case")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    seg_rows, head_rows = [], []
    cases = pd.read_csv(a.cases)
    for case, split in zip(cases.image_id, cases.split):
        gt = {}
        for b in BONES:
            gt[b], spacing = load_mask(a.data_dir / case / "segmentations" / f"{b}.nii.gz")
        gt_head = {s: femoral_head(gt[f"femur_{s}"], gt[f"hip_{s}"], spacing) for s in ("left", "right")}
        if a.stl:
            (a.out / "stl" / case).mkdir(parents=True, exist_ok=True)
            mask_to_mesh(gt["femur_left"], spacing).export(a.out / "stl" / case / "femur_left_gt.stl")

        for mode in a.modes:
            pdir = a.preds / mode / case
            if not (pdir / "femur_left.nii.gz").exists():
                print("missing prediction", mode, case)
                continue
            pr = {b: load_mask(pdir / f"{b}.nii.gz")[0] for b in BONES}
            for b in BONES:
                row = {"image_id": case, "split": split, "mode": mode, "bone": b,
                       "dice": dice(gt[b], pr[b]), **surface_metrics(gt[b], pr[b], spacing)}
                seg_rows.append(row)
            for s in ("left", "right"):
                ph = femoral_head(pr[f"femur_{s}"], pr[f"hip_{s}"], spacing)
                g = gt_head[s]
                row = {"image_id": case, "split": split, "mode": mode, "side": s,
                       "gt_diam_mm": g["diameter_mm"] if g else np.nan,
                       "pred_diam_mm": ph["diameter_mm"] if ph else np.nan,
                       "gt_fit_rmse_mm": g["rmse_mm"] if g else np.nan,
                       "centre_err_mm": (np.linalg.norm(g["centre_mm"] - ph["centre_mm"])
                                         if (g and ph) else np.nan)}
                row["diam_err_mm"] = row["pred_diam_mm"] - row["gt_diam_mm"]
                head_rows.append(row)
            if a.stl:
                mask_to_mesh(pr["femur_left"], spacing).export(a.out / "stl" / case / f"femur_left_{mode}.stl")
            print("done", case, mode)

        # Save incrementally.
        pd.DataFrame(seg_rows).to_csv(a.out / "segmentation_metrics.csv", index=False)
        pd.DataFrame(head_rows).to_csv(a.out / "femoral_head.csv", index=False)

    seg = pd.DataFrame(seg_rows)
    head = pd.DataFrame(head_rows)
    summary = seg.groupby(["mode", "bone"])[["dice", "hd95", "assd"]].agg(["mean", "std", "median"]).round(3)
    head_summary = head.groupby("mode").agg(
        n=("diam_err_mm", "count"),
        mean_diam_err=("diam_err_mm", "mean"),
        sd_diam_err=("diam_err_mm", "std"),
        mae_diam=("diam_err_mm", lambda x: x.abs().mean()),
        mean_centre_err=("centre_err_mm", "mean"),
    ).round(2)
    print(summary.to_string())
    print(head_summary.to_string())
    with open(a.out / "summary.md", "w") as f:
        f.write("## Segmentation (Dice, HD95 mm, ASSD mm)\n\n" + summary.to_markdown() + "\n\n")
        f.write("## Femoral-head diameter error (pred - GT, mm) and centre error (mm)\n\n"
                + head_summary.to_markdown() + "\n")


if __name__ == "__main__":
    main()
