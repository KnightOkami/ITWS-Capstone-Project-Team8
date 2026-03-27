import numpy as np
from sklearn.cluster import DBSCAN


def load_xyz_from_txt(file_storage) -> np.ndarray:
    """
    Load a point cloud from an uploaded .txt file.
    Accepts files with 3+ columns and keeps only XYZ.
    """
    points = np.loadtxt(file_storage).astype(np.float32)

    if points.ndim == 1:
        if points.shape[0] < 3:
            raise ValueError("Point cloud must contain at least 3 values per row (X Y Z)")
        points = np.expand_dims(points, axis=0)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Point cloud file must have at least 3 columns (X Y Z)")

    return points[:, :3]


def sample_points(points: np.ndarray, max_points: int = 115000) -> np.ndarray:
    """
    Downsample for clustering speed if the cloud is very large.
    """
    if len(points) <= max_points:
        return points

    idx = np.random.choice(len(points), max_points, replace=False)
    return points[idx]


def normalize_points(points: np.ndarray) -> np.ndarray:
    """
    Center and scale points to improve DBSCAN behavior across files.
    """
    centered = points - np.mean(points, axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale > 0:
        centered = centered / scale
    return centered


def cluster_pointcloud(points_xyz: np.ndarray, eps: float = 0.06, min_samples: int = 20):
    """
    Run DBSCAN clustering on XYZ points.
    Returns:
      - original XYZ points
      - cluster labels per point
      - object summaries
    """
    normalized = normalize_points(points_xyz)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(normalized)

    unique_labels = sorted(set(labels.tolist()))
    instances = []

    for label in unique_labels:
        if label == -1:
            continue  # noise
        mask = labels == label
        cluster_pts = points_xyz[mask]

        mins = cluster_pts.min(axis=0).tolist()
        maxs = cluster_pts.max(axis=0).tolist()

        instances.append({
            "id": int(label),
            "count": int(mask.sum()),
            "bbox_min": [float(v) for v in mins],
            "bbox_max": [float(v) for v in maxs],
        })

    return {
        "num_points": int(len(points_xyz)),
        "num_instances": int(len(instances)),
        "instances": instances,
        "points": [
            [float(p[0]), float(p[1]), float(p[2]), int(lbl)]
            for p, lbl in zip(points_xyz, labels)
        ],
    }