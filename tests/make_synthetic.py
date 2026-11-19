"""Build a tiny synthetic 'TotalSegmentator-like' dataset to test the pipeline offline.
Femoral head = sphere of known radius, so the sphere fit has a known answer."""
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd

root = Path(sys.argv[1])
SPACING = np.array([0.8, 0.8, 1.5])
SHAPE = (200, 90, 110)
R_HEAD = 23.0  # mm -> true diameter 46 mm


def grid():
    idx = np.indices(SHAPE).reshape(3, -1).T * SPACING
    return idx.reshape(*SHAPE, 3)


def side(P, cx, sign):
    c = np.array([cx, 36.0, 100.0])
    d = np.linalg.norm(P - c, axis=-1)
    head = d <= R_HEAD
    # neck: cylinder from head centre going lateral (sign) and down
    axis = np.array([sign * 0.7, 0.0, -0.7]); axis /= np.linalg.norm(axis)
    v = P - c
    t = v @ axis
    radial = np.linalg.norm(v - t[..., None] * axis, axis=-1)
    neck = (t > 0) & (t < 45) & (radial < 13)
    # shaft: vertical cylinder below the neck end
    s_c = c + 45 * axis
    shaft = (np.linalg.norm((P - s_c)[..., :2], axis=-1) < 12) & (P[..., 2] < s_c[2])
    femur = head | neck | shaft
    # acetabulum: shell around the head covering the upper/medial part, plus a block above
    shell = (d > R_HEAD + 2.5) & (d < R_HEAD + 10) & (P[..., 2] > c[2] - 4)
    block = (np.abs(P[..., 0] - cx) < 30) & (P[..., 2] > c[2] + R_HEAD + 8) & (P[..., 2] < c[2] + R_HEAD + 30)
    hip = (shell | block) & ~femur
    return femur, hip


def main():
    P = grid()
    affine = np.diag([*SPACING, 1.0])
    rng = np.random.default_rng(0)
    rows = []
    for case, split in [("s0001", "test"), ("s0002", "train"), ("s0003", "test")]:
        d = root / case / "segmentations"
        d.mkdir(parents=True, exist_ok=True)
        fl, hl = side(P, 50.0, +1)
        fr, hr = side(P, 110.0, -1)
        ct = rng.normal(0, 30, SHAPE).astype(np.float32)
        ct[fl | fr] = 900
        ct[hl | hr] = 700
        nib.save(nib.Nifti1Image(ct, affine), str(root / case / "ct.nii.gz"))
        for name, m in [("femur_left", fl), ("femur_right", fr), ("hip_left", hl), ("hip_right", hr)]:
            nib.save(nib.Nifti1Image(m.astype(np.uint8), affine), str(d / f"{name}.nii.gz"))
        rows.append({"image_id": case, "split": split})
    pd.DataFrame(rows).to_csv(root / "meta.csv", sep=";", index=False)
    print("synthetic dataset at", root)


if __name__ == "__main__":
    main()
