"""Per-hip quality checks, run after evaluate.py. Writes three CSVs to the results folder.

1. implant_flags.csv  - metal near the femur: more than --metal_voxels voxels above --metal_hu HU
                        within 3 voxels of the ground-truth femur mask (a hip replacement).
2. head_fov_flags.csv - whether the ground-truth femoral head extends below the bottom slice
                        of the scan (head cut off, so the sphere fit is not comparable).
3. pred_fit_qc.csv    - femoral-head fit details for GT and every model: diameter, fit RMSE,
                        number of surface points used, and number of connected pieces in each mask.

These checks were added during the study, after the first implant case (s0058) was seen.
They are applied identically to every hip, but they were not specified in advance.

Usage: python scripts/quality_checks.py DATA_DIR cases.csv --preds preds --out results
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from femur3d.io import load, load_mask  # noqa: E402
from femur3d.geometry import femoral_head  # noqa: E402

SIDES = ("left", "right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("cases", type=Path)
    ap.add_argument("--preds", type=Path, default=Path("preds"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--modes", nargs="+", default=["full", "fast"])
    ap.add_argument("--metal_hu", type=float, default=2500)
    ap.add_argument("--metal_voxels", type=int, default=200)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    imp, fov, fit = [], [], []
    for case in pd.read_csv(a.cases).image_id:
        gt_dir = a.data_dir / case / "segmentations"
        ct, _ = load(a.data_dir / case / "ct.nii.gz", np.float32)
        for s in SIDES:
            femur, spacing = load_mask(gt_dir / f"femur_{s}.nii.gz")
            n = int((ct[ndimage.binary_dilation(femur, iterations=3)] > a.metal_hu).sum()) if femur.any() else 0
            imp.append({"image_id": case, "side": s, "metal_voxels": n, "implant": n > a.metal_voxels})

            for src in ["gt"] + a.modes:
                d = gt_dir if src == "gt" else a.preds / src / case
                f, sp = load_mask(d / f"femur_{s}.nii.gz")
                h, _ = load_mask(d / f"hip_{s}.nii.gz")
                r = femoral_head(f, h, sp)
                fit.append({"image_id": case, "side": s, "source": src,
                            "diam_mm": r["diameter_mm"] if r else np.nan,
                            "fit_rmse_mm": r["rmse_mm"] if r else np.nan,
                            "n_points": r["n_points"] if r else 0,
                            "femur_parts": ndimage.label(f)[1], "hip_parts": ndimage.label(h)[1]})
                if src == "gt":
                    # Distance (mm) from the lowest point of the head sphere to the bottom slice (z = 0 in RAS).
                    below = float(r["centre_mm"][2] - r["diameter_mm"] / 2) if r else np.nan
                    fov.append({"image_id": case, "side": s, "head_below_mm": round(below, 1),
                                "head_truncated": not (below >= 1.5)})
        print("checked", case)

    pd.DataFrame(imp).to_csv(a.out / "implant_flags.csv", index=False)
    pd.DataFrame(fov).to_csv(a.out / "head_fov_flags.csv", index=False)
    pd.DataFrame(fit).to_csv(a.out / "pred_fit_qc.csv", index=False)
    flagged = pd.DataFrame(imp).merge(pd.DataFrame(fov), on=["image_id", "side"])
    print(flagged[flagged.implant | flagged.head_truncated].to_string(index=False))


if __name__ == "__main__":
    main()
