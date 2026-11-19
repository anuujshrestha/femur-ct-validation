"""3D geometry: mask -> STL mesh, and femoral-head sphere fitting.

Femoral-head size is a planning input in hip arthroplasty (head/cup sizing)
and the hip centre is one end of the femoral mechanical axis.
Here it is estimated from the segmentation alone, so errors in the
segmentation propagate into a clinically meaningful number.
"""
import numpy as np
from scipy import ndimage
from skimage import measure
import trimesh


def mask_to_mesh(mask: np.ndarray, spacing, smooth_iters: int = 10) -> trimesh.Trimesh:
    """Marching cubes on a binary mask; vertices in mm (grid coordinates * spacing)."""
    padded = np.pad(mask.astype(np.uint8), 1)  # pad so the surface closes at volume edges
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5, spacing=tuple(spacing))
    verts -= np.asarray(spacing)  # undo the 1-voxel pad offset
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if mesh.volume < 0:  # marching-cubes winding depends on axis order; make normals point outward
        mesh.invert()
    if smooth_iters > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iters)  # volume-preserving smoothing
    return mesh


def surface_points(mask: np.ndarray, spacing) -> np.ndarray:
    """Sub-voxel surface points (mm) from marching cubes at the 0.5 iso-level.
    Voxel centres of boundary voxels sit ~half a voxel inside the true surface and
    bias the fitted radius low; marching-cubes vertices do not."""
    sl = _bbox(mask, margin=2)
    offset = np.array([s.start for s in sl]) * np.asarray(spacing)
    verts, _, _, _ = measure.marching_cubes(np.pad(mask[sl].astype(np.uint8), 1), level=0.5,
                                            spacing=tuple(spacing))
    return verts - np.asarray(spacing) + offset


def _bbox(mask, margin=0):
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, mask.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def fit_sphere(points: np.ndarray):
    """Algebraic least-squares sphere fit. points: (N, 3) in mm. Returns (centre, radius, rmse)."""
    A = np.c_[2 * points, np.ones(len(points))]
    b = (points ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    centre = sol[:3]
    radius = np.sqrt(sol[3] + centre @ centre)
    resid = np.linalg.norm(points - centre, axis=1) - radius
    return centre, float(radius), float(np.sqrt(np.mean(resid ** 2)))


def femoral_head(femur: np.ndarray, hip: np.ndarray, spacing,
                 contact_mm: float = 6.0, band_mm: float = 2.0, n_refine: int = 3,
                 min_points: int = 50):
    """Estimate femoral-head centre and diameter from femur + hip-bone masks.

    1. Femur surface voxels within `contact_mm` of the hip bone = the part of the
       head that faces the acetabulum (joint space is only a few mm).
    2. Fit a sphere to those points.
    3. Refine: refit using all femur surface points within `band_mm` of the sphere,
       which pulls in the rest of the head but not the narrower neck.
    Returns dict (centre_mm, diameter_mm, rmse_mm, n_points) or None on failure.
    """
    femur, hip = femur.astype(bool), hip.astype(bool)
    if femur.sum() == 0 or hip.sum() == 0:
        return None
    spacing = np.asarray(spacing, float)
    # Restrict work to the region around the hip bone (the head lies next to it).
    sl = _bbox(hip, margin=int(np.ceil(60 / spacing.min())))
    offset = np.array([s.start for s in sl]) * spacing
    femur_c, hip_c = femur[sl], hip[sl]
    if femur_c.sum() == 0:
        return None
    dist_to_hip = ndimage.distance_transform_edt(~hip_c, sampling=spacing)
    surf_pts = surface_points(femur_c, spacing)                      # mm, local to the crop
    vox = np.clip(np.rint(surf_pts / spacing).astype(int), 0, np.array(femur_c.shape) - 1)
    d_hip = dist_to_hip[vox[:, 0], vox[:, 1], vox[:, 2]]
    contact = surf_pts[d_hip < contact_mm]
    if len(contact) < min_points:
        return None
    centre, r, rmse = fit_sphere(contact)
    pts = contact
    for _ in range(n_refine):
        d = np.linalg.norm(surf_pts - centre, axis=1)
        sel = surf_pts[np.abs(d - r) < band_mm]
        if len(sel) < min_points:
            break
        centre, r, rmse = fit_sphere(sel)
        pts = sel
    return {"centre_mm": centre + offset, "diameter_mm": 2 * r, "rmse_mm": rmse, "n_points": int(len(pts))}
