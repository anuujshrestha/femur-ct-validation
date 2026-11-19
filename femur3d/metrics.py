"""Overlap and surface-distance metrics for binary 3D masks.

All distances are in millimetres: voxel spacing is passed in explicitly,
so anisotropic CT voxels are handled correctly.
"""
import numpy as np
from scipy import ndimage


def dice(gt: np.ndarray, pred: np.ndarray) -> float:
    """Dice similarity coefficient. Returns nan if both masks are empty."""
    gt, pred = gt.astype(bool), pred.astype(bool)
    denom = gt.sum() + pred.sum()
    if denom == 0:
        return float("nan")
    return 2.0 * np.logical_and(gt, pred).sum() / denom


def _surface(mask: np.ndarray) -> np.ndarray:
    """Boundary voxels: voxels in the mask that touch the background (6-connectivity)."""
    eroded = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1))
    return np.logical_and(mask, ~eroded)


def _directed_distances(src_surf, dst_surf, spacing):
    """Distance (mm) from every surface voxel of src to the nearest surface voxel of dst."""
    # EDT of the complement of dst's surface gives, at each voxel, distance to dst's surface.
    dt = ndimage.distance_transform_edt(~dst_surf, sampling=spacing)
    return dt[src_surf]


def surface_metrics(gt: np.ndarray, pred: np.ndarray, spacing) -> dict:
    """HD95 and ASSD in mm.

    HD95 follows the MONAI convention: max of the two directed 95th percentiles.
    ASSD = mean of all symmetric surface distances.
    Returns nan values if either mask is empty.
    """
    gt, pred = gt.astype(bool), pred.astype(bool)
    if gt.sum() == 0 or pred.sum() == 0:
        return {"hd95": float("nan"), "assd": float("nan")}
    # Crop to the joint bounding box (plus margin) for speed.
    sl = _joint_bbox(gt | pred, margin=3)
    gt, pred = gt[sl], pred[sl]
    s_gt, s_pr = _surface(gt), _surface(pred)
    d_gp = _directed_distances(s_gt, s_pr, spacing)
    d_pg = _directed_distances(s_pr, s_gt, spacing)
    hd95 = max(np.percentile(d_gp, 95), np.percentile(d_pg, 95))
    assd = np.concatenate([d_gp, d_pg]).mean()
    return {"hd95": float(hd95), "assd": float(assd)}


def _joint_bbox(mask, margin=0):
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, mask.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))
