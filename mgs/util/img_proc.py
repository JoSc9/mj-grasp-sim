from typing import Tuple

import numpy as np


def voxel_downsample_pcd(
    points: np.ndarray, features: np.ndarray, voxel_size: float, segmentation_label=None
) -> Tuple[np.ndarray, np.ndarray]:
    if segmentation_label is not None:
        if points.shape[0]  != segmentation_label.shape[0]:
            raise ValueError(
                "points and segmentation_label must have the same number of points"
            ) 
    
    
    mins = np.min(points, axis=0)  # Shape: (3,)
    vox_idx = np.floor_divide(points - mins, voxel_size).astype(
        np.int64
    )  # Shape: (N, 3)
    shape = np.max(vox_idx, axis=0) + 1  # Shape: (3,)
    raveled_idx = np.ravel_multi_index(vox_idx.T, shape)  # Shape: (N,)
    n_voxels = np.prod(shape)
    n_pts_per_vox = np.bincount(raveled_idx, minlength=n_voxels)  # Shape: (n_voxels,)
    nonzero_vox = np.nonzero(n_pts_per_vox)[0]  # Shape: (num_nonzero_voxels,)
    # Shape: (num_nonzero_voxels,)
    n_pts_per_vox_nonzero = n_pts_per_vox[nonzero_vox]
    feature_sum = np.zeros(
        (n_voxels, features.shape[1]), dtype=features.dtype
    )  # Shape: (n_voxels, C)
    np.add.at(feature_sum, raveled_idx, features)
    feature_vox = feature_sum[nonzero_vox]  # Shape: (num_nonzero_voxels, C)
    coord_sum = np.zeros(
        (n_voxels, points.shape[1]), dtype=points.dtype
    )  # Shape: (n_voxels, 3)
    np.add.at(coord_sum, raveled_idx, points)
    coord_vox = coord_sum[nonzero_vox]  # Shape: (num_nonzero_voxels, 3)
    n_pts_per_vox_nonzero = n_pts_per_vox_nonzero[
        :, np.newaxis
    ]  # Shape: (num_nonzero_voxels, 1)
    feature_vox /= n_pts_per_vox_nonzero
    coord_vox /= n_pts_per_vox_nonzero
    # If segmentation labels were provided, compute a label per voxel.
    if segmentation_label is not None:
        seg = np.asarray(segmentation_label)
        # Vectorized majority-vote per non-empty voxel.
        # Group points by their raveled voxel index into contiguous groups
        uniq_vox, inv = np.unique(raveled_idx, return_inverse=True)

        # Remap labels to 0..L-1 so we can use a single bincount
        uniq_labels, labels_inv = np.unique(seg, return_inverse=True)
        L = uniq_labels.size
        M = uniq_vox.shape[0]

        # Combined index: for each point, group_index * L + label_index
        comb = inv * L + labels_inv
        counts_gl = np.bincount(comb, minlength=M * L).reshape(M, L)
        majority_idx = counts_gl.argmax(axis=1)
        label_vox = uniq_labels[majority_idx]

        # Ensure ordering matches `nonzero_vox` (should usually be equal)
        if not np.array_equal(uniq_vox, nonzero_vox):
            pos = np.searchsorted(nonzero_vox, uniq_vox)
            reordered = np.empty(nonzero_vox.shape[0], dtype=label_vox.dtype)
            reordered[pos] = label_vox
            label_vox = reordered

        return coord_vox, feature_vox, label_vox

    return coord_vox, feature_vox


def rgbd_to_pcd(rgbd, intrinsics, extrinsics) -> Tuple[np.ndarray, np.ndarray]:
    """
    rgbd: numpy array of shape (N, height, width, k) where the last channel is the depth value
    intrinsics: numpy array of shape (3, 3) representing the camera intrinsics matrix
    extrinsics: numpy array of shape (N, 4, 4) representing the camera extrinsics matrix
    """
    width, height = rgbd.shape[1], rgbd.shape[2]
    fx, fy, cx, cy = (
        intrinsics[0, 0],
        intrinsics[1, 1],
        intrinsics[0, 2],
        intrinsics[1, 2],
    )
    z = rgbd[..., -1]
    u = np.arange(width) - cx
    v = np.arange(height) - cy
    x = (z * u) / fx
    y = np.transpose((np.transpose(z, axes=[0, 2, 1]) * v), axes=[0, 2, 1]) / fy

    points = np.stack((x, y, z), axis=-1)
    points_homo = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)
    points_homo = np.einsum("nij,nhwj->nhwi", extrinsics, points_homo)
    points = points_homo[..., :3]
    color = rgbd[..., :-1]
    return (points, color)


def detect_outlier(
    points: np.ndarray, radius: float, min_neighbors: int = 8
) -> np.ndarray:
    """
    Radius-based outlier removal using a uniform grid spatial hash (NumPy only).

    Args:
        points: (N, 3) float array of 3D points.
        radius: neighborhood radius (same units as points).
        min_neighbors: keep a point if it has at least this many neighbors
                       within 'radius' (excluding itself).

    Returns:
        mask: (N,) boolean array; True = inlier, False = outlier.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an (N, 3) array")
    n = points.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    r2 = float(radius) * float(radius)
    inv = 1.0 / float(radius)

    # Hash each point to a grid cell
    grid_idx = np.floor(points * inv).astype(np.int32)
    cell_keys = [tuple(ix) for ix in grid_idx]

    # Build cell -> list of point indices
    cells: dict[tuple, list] = {}
    for i, key in enumerate(cell_keys):
        cells.setdefault(key, []).append(i)

    # Offsets for the 27 neighboring cells (including the cell itself)
    offsets = np.array(
        [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
        dtype=np.int32,
    )

    mask = np.zeros(n, dtype=bool)
    pts = points  # alias for speed

    for i, key in enumerate(cell_keys):
        base = np.fromiter(key, dtype=np.int32, count=3)
        count = 0

        # Scan neighboring cells; break early once threshold is met
        for off in offsets:
            k = tuple((base + off).tolist())
            idxs = cells.get(k)
            if not idxs:
                continue

            cand = pts[idxs]
            # squared distances to candidates
            d2 = np.sum((cand - pts[i]) * (cand - pts[i]), axis=1)
            # within radius
            count += np.count_nonzero(d2 <= r2)
            if count - 1 >= min_neighbors:  # minus self, counted once
                mask[i] = True
                break

        if not mask[i]:
            # finalize with self excluded (self is always in its own cell)
            if count - 1 >= min_neighbors:
                mask[i] = True

    return mask
