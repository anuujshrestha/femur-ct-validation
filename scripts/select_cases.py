"""Pick CT cases whose ground truth contains both femoral heads and both hip bones.

Prefers the official test split (meta.csv 'split' column) because TotalSegmentator
was trained on the train split; evaluating on training cases would be optimistic.

Usage: python scripts/select_cases.py DATA_DIR --n 15 --out cases.csv
"""
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd

STRUCTS = ["femur_left", "femur_right", "hip_left", "hip_right"]


def has_structures(case_dir: Path, min_voxels: int) -> bool:
    for s in STRUCTS:
        p = case_dir / "segmentations" / f"{s}.nii.gz"
        if not p.exists():
            return False
        if np.count_nonzero(np.asanyarray(nib.load(str(p)).dataobj)) < min_voxels:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--min_voxels", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=Path("cases.csv"))
    a = ap.parse_args()

    meta_path = a.data_dir / "meta.csv"
    meta = None
    if meta_path.exists():
        # utf-8-sig drops an invisible byte-order mark; columns are normalised because the
        # real meta.csv could not be matched on 'image_id' without this.
        meta = pd.read_csv(meta_path, sep=None, engine="python", encoding="utf-8-sig")
        meta.columns = [str(c).strip().lower() for c in meta.columns]
        print("meta.csv columns:", list(meta.columns))
        if "image_id" not in meta.columns:
            id_col = next((c for c in meta.columns if "id" in c or "subject" in c), meta.columns[0])
            print(f"no 'image_id' column; using '{id_col}' as the case ID")
            meta = meta.rename(columns={id_col: "image_id"})
        meta["image_id"] = meta["image_id"].astype(str).str.strip()
        if "split" in meta.columns:
            meta["split"] = meta["split"].astype(str).str.strip().str.lower()
    cases = sorted(p.name for p in a.data_dir.iterdir() if (p / "ct.nii.gz").exists())
    df = pd.DataFrame({"image_id": cases})
    if meta is not None and "split" in meta.columns:
        df = df.merge(meta[["image_id", "split"]], on="image_id", how="left")
    else:
        df["split"] = "unknown"
    print(df["split"].value_counts(dropna=False).to_string())

    # Test cases first, then validation, then train (flagged in the output).
    order = {"test": 0, "val": 1, "validation": 1, "train": 2}
    df["rank"] = df["split"].map(order).fillna(3)
    df = df.sort_values(["rank", "image_id"])

    keep = []
    for _, row in df.iterrows():
        if has_structures(a.data_dir / row.image_id, a.min_voxels):
            keep.append(row)
            print("keep", row.image_id, row.split)
        if len(keep) >= a.n:
            break
    out = pd.DataFrame(keep)[["image_id", "split"]]
    out.to_csv(a.out, index=False)
    print(f"{len(out)} cases written to {a.out}; splits: {out.split.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
