"""NIfTI loading. Everything is reoriented to the closest canonical (RAS) orientation
so that CT, ground truth and predictions share axes: 0 = left-right,
1 = posterior-anterior, 2 = inferior-superior."""
import nibabel as nib
import numpy as np


def load(path, dtype=None):
    img = nib.as_closest_canonical(nib.load(str(path)))
    arr = np.asanyarray(img.dataobj)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr, tuple(float(z) for z in img.header.get_zooms()[:3])


def load_mask(path):
    arr, spacing = load(path)
    return arr > 0, spacing
