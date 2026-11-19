"""Run TotalSegmentator (normal 1.5 mm and --fast 3 mm models) on the selected cases.

Usage: python scripts/run_totalseg.py DATA_DIR cases.csv --out preds
Writes preds/<mode>/<case>/<structure>.nii.gz and preds/runtimes.csv
"""
import argparse
import subprocess
import time
from pathlib import Path
import pandas as pd

ROI = ["femur_left", "femur_right", "hip_left", "hip_right"]
MODES = {"full": [], "fast": ["--fast"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("cases", type=Path)
    ap.add_argument("--out", type=Path, default=Path("preds"))
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    a = ap.parse_args()

    log = a.out / "runtimes.csv"
    rows = pd.read_csv(log).to_dict("records") if log.exists() else []  # keep earlier runs
    for case in pd.read_csv(a.cases).image_id:
        for mode in a.modes:
            out_dir = a.out / mode / case
            if (out_dir / "femur_left.nii.gz").exists():
                print("skip (done)", mode, case)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            a.out.mkdir(parents=True, exist_ok=True)
            cmd = ["TotalSegmentator", "-i", str(a.data_dir / case / "ct.nii.gz"),
                   "-o", str(out_dir), "--roi_subset", *ROI, *MODES[mode]]
            t0 = time.time()
            res = subprocess.run(cmd, capture_output=True, text=True)
            dt = time.time() - t0
            ok = res.returncode == 0
            print(f"{mode:4s} {case}: {'ok' if ok else 'FAILED'} in {dt:.1f}s")
            if not ok:
                print(res.stderr[-2000:])
            rows.append({"image_id": case, "mode": mode, "seconds": round(dt, 1), "ok": ok})
            # Append after every run so a Colab disconnect loses nothing.
            pd.DataFrame(rows).to_csv(log, index=False)


if __name__ == "__main__":
    main()
