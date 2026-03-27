import os
import sys
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

POINTNET2_REPO = os.getenv(
    "POINTNET2_REPO",
    "/app/third_party/Pointnet_Pointnet2_pytorch"
)

POINTNET2_CKPT = os.getenv(
    "POINTNET2_CKPT",
    "/app/checkpoints/pointnet2_sem_seg/best_model.pth"
)

NUM_CLASSES = 13
NUM_POINT = 2048
BLOCK_SIZE = 1.5
STRIDE = 0.75
BATCH_SIZE = 2

# Safe defaults for local CPU Docker dev
MAX_INPUT_POINTS = 200000
MAX_RETURN_POINTS = 50000

CLASS_NAMES = [
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "chair",
    "table",
    "bookcase",
    "sofa",
    "board",
    "clutter",
]

# Large structural classes usually do not need DBSCAN instance extraction
STRUCTURAL_CLASS_IDS = {0, 1, 2, 3, 4}  # ceiling, floor, wall, beam, column


def load_points_and_optional_rgb(file_storage):
    points = np.loadtxt(file_storage).astype(np.float32)

    if points.ndim == 1:
        if points.shape[0] < 3:
            raise ValueError("Point cloud must contain at least 3 values per row")
        points = np.expand_dims(points, axis=0)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Point cloud must have at least 3 columns (XYZ)")

    xyz = points[:, :3]

    if points.shape[1] >= 6:
        rgb = points[:, 3:6].copy()
    else:
        rgb = np.zeros((points.shape[0], 3), dtype=np.float32)

    return xyz, rgb


def normalize_rgb(rgb):
    if rgb.size == 0:
        return rgb
    max_val = np.max(rgb)
    if max_val > 1.0:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def room_normalized_xyz(xyz):
    room_min = xyz.min(axis=0)
    room_max = xyz.max(axis=0)
    room_span = np.maximum(room_max - room_min, 1e-6)
    return (xyz - room_min) / room_span, room_min, room_max


def cap_input_points(xyz, rgb, max_input_points=MAX_INPUT_POINTS):
    if len(xyz) <= max_input_points:
        return xyz, rgb

    idx = np.random.choice(len(xyz), max_input_points, replace=False)
    return xyz[idx], rgb[idx]


def build_block_coords(xyz, block_size=BLOCK_SIZE, stride=STRIDE):
    coord_min = xyz.min(axis=0)
    coord_max = xyz.max(axis=0)

    xbeg_list = []
    ybeg_list = []

    x = coord_min[0]
    while x <= coord_max[0]:
        y = coord_min[1]
        while y <= coord_max[1]:
            xbeg_list.append(x)
            ybeg_list.append(y)
            y += stride
        x += stride

    return list(zip(xbeg_list, ybeg_list))


def make_block_features(xyz, rgb, room_xyz_norm, chosen_idxs, xbeg, ybeg, block_size=BLOCK_SIZE):
    block_xyz = xyz[chosen_idxs].copy()
    block_rgb = rgb[chosen_idxs].copy()
    block_xyz_norm = room_xyz_norm[chosen_idxs].copy()

    center_x = xbeg + block_size / 2.0
    center_y = ybeg + block_size / 2.0

    # Local XY centering for block inference
    block_xyz[:, 0] -= center_x
    block_xyz[:, 1] -= center_y

    # 9D features expected by sem-seg model:
    # centered xyz + rgb + room-normalized xyz
    features = np.concatenate([block_xyz, block_rgb, block_xyz_norm], axis=1).astype(np.float32)
    return features


def iter_room_block_batches(
    xyz,
    rgb,
    num_point=NUM_POINT,
    block_size=BLOCK_SIZE,
    stride=STRIDE,
    batch_size=BATCH_SIZE,
):
    """
    Generator that yields small inference batches instead of building all blocks
    in memory at once.
    """
    rgb = normalize_rgb(rgb)
    room_xyz_norm, _, _ = room_normalized_xyz(xyz)
    block_coords = build_block_coords(xyz, block_size=block_size, stride=stride)

    batch_features = []
    batch_point_indices = []

    yielded_any = False

    for xbeg, ybeg in block_coords:
        xcond = (xyz[:, 0] >= xbeg) & (xyz[:, 0] <= xbeg + block_size)
        ycond = (xyz[:, 1] >= ybeg) & (xyz[:, 1] <= ybeg + block_size)
        point_idxs = np.where(xcond & ycond)[0]

        if len(point_idxs) < 64:
            continue

        yielded_any = True

        if len(point_idxs) >= num_point:
            chosen = np.random.choice(point_idxs, num_point, replace=False)
        else:
            chosen = np.random.choice(point_idxs, num_point, replace=True)

        features = make_block_features(
            xyz, rgb, room_xyz_norm, chosen, xbeg, ybeg, block_size=block_size
        )

        batch_features.append(features)
        batch_point_indices.append(chosen)

        if len(batch_features) == batch_size:
            yield np.stack(batch_features, axis=0), batch_point_indices
            batch_features = []
            batch_point_indices = []

    if batch_features:
        yield np.stack(batch_features, axis=0), batch_point_indices

    # Fallback if no spatial blocks were usable
    if not yielded_any:
        if len(xyz) >= num_point:
            chosen = np.random.choice(len(xyz), num_point, replace=False)
        else:
            chosen = np.random.choice(len(xyz), num_point, replace=True)

        room_xyz_norm, _, _ = room_normalized_xyz(xyz)
        center = np.mean(xyz[chosen][:, :2], axis=0)

        block_xyz = xyz[chosen].copy()
        block_xyz[:, 0] -= center[0]
        block_xyz[:, 1] -= center[1]

        features = np.concatenate(
            [block_xyz, normalize_rgb(rgb[chosen].copy()), room_xyz_norm[chosen].copy()],
            axis=1
        ).astype(np.float32)

        yield np.expand_dims(features, axis=0), [chosen]


def apply_structural_height_prior(xyz, semantic_ids):
    """
    Very light geometric cleanup:
    - only correct obviously impossible ceiling/floor labels
    - do NOT aggressively force points into wall
    """
    z = xyz[:, 2]
    z_min = np.min(z)
    z_max = np.max(z)
    z_span = max(z_max - z_min, 1e-6)
    z_norm = (z - z_min) / z_span

    adjusted = semantic_ids.copy()

    for i in range(len(adjusted)):
        # ceiling too low -> wall
        if adjusted[i] == 0 and z_norm[i] < 0.80:
            adjusted[i] = 2

        # floor too high -> wall
        elif adjusted[i] == 1 and z_norm[i] > 0.20:
            adjusted[i] = 2

        # DO NOT force wall into ceiling/floor unless extremely obvious
        elif adjusted[i] == 2:
            if z_norm[i] > 0.98:
                adjusted[i] = 0
            elif z_norm[i] < 0.02:
                adjusted[i] = 1

    return adjusted

def smooth_semantic_labels(xyz, semantic_ids, semantic_scores, k=16):
    """
    Softer smoothing:
    - only relabel when neighborhood majority is strong
    - avoid collapsing everything into wall
    """
    if len(xyz) <= k:
        return semantic_ids

    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nbrs.fit(xyz)
    _, indices = nbrs.kneighbors(xyz)

    smoothed = semantic_ids.copy()

    for i in range(len(xyz)):
        neighbor_labels = semantic_ids[indices[i]]
        counts = np.bincount(neighbor_labels, minlength=NUM_CLASSES)
        majority = np.argmax(counts)

        # require stronger majority before changing label
        if majority != semantic_ids[i]:
            majority_ratio = counts[majority] / k
            current_conf = float(np.max(semantic_scores[i]))

            majority_mask = neighbor_labels == majority
            if np.any(majority_mask):
                majority_neighbor_idxs = indices[i][majority_mask]
                majority_conf = float(
                    np.mean(np.max(semantic_scores[majority_neighbor_idxs], axis=1))
                )
            else:
                majority_conf = 0.0

            if majority_ratio >= 0.75 and majority_conf >= current_conf * 0.95:
                smoothed[i] = majority

    return smoothed

class PointNet2SemanticSegmenter:
    def __init__(self):
        if not os.path.isdir(POINTNET2_REPO):
            raise RuntimeError(f"PointNet++ repo not found at {POINTNET2_REPO}")

        if not os.path.isfile(POINTNET2_CKPT):
            raise RuntimeError(f"Checkpoint not found at {POINTNET2_CKPT}")

        if POINTNET2_REPO not in sys.path:
            sys.path.insert(0, POINTNET2_REPO)

        from models.pointnet2_sem_seg import get_model  # noqa

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(NUM_CLASSES).to(self.device)
        self.model.eval()

        try:
            checkpoint = torch.load(
                POINTNET2_CKPT,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                POINTNET2_CKPT,
                map_location=self.device,
            )

        if isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get("model_state_dict")
                or checkpoint.get("state_dict")
                or checkpoint
            )
        else:
            state_dict = checkpoint

        cleaned = {}
        for key, value in state_dict.items():
            cleaned[key.replace("module.", "")] = value

        self.model.load_state_dict(cleaned, strict=True)

    @torch.no_grad()
    def predict_semantics(self, xyz, rgb):
        score_accum = np.zeros((len(xyz), NUM_CLASSES), dtype=np.float32)
        vote_accum = np.zeros((len(xyz), 1), dtype=np.float32)

        print(f"[segment] begin chunked inference on {len(xyz)} points", flush=True)

        batch_counter = 0
        for batch_features, batch_point_indices in iter_room_block_batches(
            xyz,
            rgb,
            num_point=NUM_POINT,
            block_size=BLOCK_SIZE,
            stride=STRIDE,
            batch_size=BATCH_SIZE,
        ):
            batch = torch.from_numpy(batch_features).permute(0, 2, 1).float().to(self.device)
            pred, _ = self.model(batch)
            probs = torch.exp(pred).cpu().numpy()  # B x N x C

            for local_block_idx, original_idxs in enumerate(batch_point_indices):
                block_probs = probs[local_block_idx]
                for j, original_idx in enumerate(original_idxs):
                    score_accum[original_idx] += block_probs[j]
                    vote_accum[original_idx] += 1.0

            batch_counter += 1
            if batch_counter % 20 == 0:
                print(f"[segment] processed {batch_counter} block batches", flush=True)

        vote_accum = np.maximum(vote_accum, 1.0)
        avg_scores = score_accum / vote_accum
        semantic_ids = np.argmax(avg_scores, axis=1)

        semantic_ids = apply_structural_height_prior(xyz, semantic_ids)
        semantic_ids = smooth_semantic_labels(xyz, semantic_ids, avg_scores, k=24)

        print("[segment] finished model inference", flush=True)
        return semantic_ids, avg_scores


def semantic_instances_from_points(xyz, semantic_ids):
    """
    Extract instances from semantic labels with much less fragmentation.
    Large structural classes become one region.
    Object-like classes are clustered, but tiny fragments are merged away.
    """
    instance_ids = np.full(len(xyz), -1, dtype=np.int32)
    instances = []
    next_instance_id = 0

    unique_semantic = sorted(set(int(v) for v in semantic_ids.tolist()))

    for semantic_id in unique_semantic:
        class_mask = semantic_ids == semantic_id
        class_xyz = xyz[class_mask]
        class_global_idx = np.where(class_mask)[0]

        if len(class_xyz) == 0:
            continue

        semantic_label = (
            CLASS_NAMES[semantic_id]
            if semantic_id < len(CLASS_NAMES)
            else f"class_{semantic_id}"
        )

        # Treat structural classes as one large semantic region
        if semantic_id in STRUCTURAL_CLASS_IDS:
            mins = class_xyz.min(axis=0).tolist()
            maxs = class_xyz.max(axis=0).tolist()

            instance_ids[class_global_idx] = next_instance_id
            instances.append({
                "id": int(next_instance_id),
                "semantic_id": int(semantic_id),
                "semantic_label": semantic_label,
                "count": int(len(class_global_idx)),
                "bbox_min": [float(v) for v in mins],
                "bbox_max": [float(v) for v in maxs],
            })
            next_instance_id += 1
            continue

        # Normalize class points for clustering
        class_min = class_xyz.min(axis=0)
        class_max = class_xyz.max(axis=0)
        class_span = np.maximum(class_max - class_min, 1e-6)
        class_xyz_norm = (class_xyz - class_min) / class_span

        cluster_labels = DBSCAN(eps=0.20, min_samples=200).fit_predict(class_xyz_norm)
        unique_clusters = sorted(set(int(v) for v in cluster_labels.tolist()))

        kept_any = False
        leftover_idx = []

        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue

            local_mask = cluster_labels == cluster_id
            member_idx = class_global_idx[local_mask]

            if len(member_idx) < 400:
                leftover_idx.extend(member_idx.tolist())
                continue

            kept_any = True
            member_xyz = xyz[member_idx]
            instance_ids[member_idx] = next_instance_id

            mins = member_xyz.min(axis=0).tolist()
            maxs = member_xyz.max(axis=0).tolist()

            instances.append({
                "id": int(next_instance_id),
                "semantic_id": int(semantic_id),
                "semantic_label": semantic_label,
                "count": int(len(member_idx)),
                "bbox_min": [float(v) for v in mins],
                "bbox_max": [float(v) for v in maxs],
            })
            next_instance_id += 1

        noise_idx = class_global_idx[cluster_labels == -1]
        if len(noise_idx) > 0:
            leftover_idx.extend(noise_idx.tolist())

        if leftover_idx:
            leftover_idx = np.array(sorted(set(leftover_idx)), dtype=np.int32)

            if len(leftover_idx) >= 400:
                kept_any = True
                leftover_xyz = xyz[leftover_idx]
                instance_ids[leftover_idx] = next_instance_id

                mins = leftover_xyz.min(axis=0).tolist()
                maxs = leftover_xyz.max(axis=0).tolist()

                instances.append({
                    "id": int(next_instance_id),
                    "semantic_id": int(semantic_id),
                    "semantic_label": semantic_label,
                    "count": int(len(leftover_idx)),
                    "bbox_min": [float(v) for v in mins],
                    "bbox_max": [float(v) for v in maxs],
                })
                next_instance_id += 1

        if not kept_any:
            mins = class_xyz.min(axis=0).tolist()
            maxs = class_xyz.max(axis=0).tolist()

            instance_ids[class_global_idx] = next_instance_id
            instances.append({
                "id": int(next_instance_id),
                "semantic_id": int(semantic_id),
                "semantic_label": semantic_label,
                "count": int(len(class_global_idx)),
                "bbox_min": [float(v) for v in mins],
                "bbox_max": [float(v) for v in maxs],
            })
            next_instance_id += 1

    return instance_ids, instances


_segmenter = None


def get_segmenter():
    global _segmenter
    if _segmenter is None:
        _segmenter = PointNet2SemanticSegmenter()
    return _segmenter


def run_ml_segmentation(file_storage):
    xyz, rgb = load_points_and_optional_rgb(file_storage)
    xyz, rgb = cap_input_points(xyz, rgb, max_input_points=MAX_INPUT_POINTS)

    print(f"[segment] using {len(xyz)} input points after cap", flush=True)

    segmenter = get_segmenter()
    semantic_ids, semantic_scores = segmenter.predict_semantics(xyz, rgb)

    print("[segment] starting instance extraction", flush=True)
    instance_ids, instances = semantic_instances_from_points(xyz, semantic_ids)
    print("[segment] finished instance extraction", flush=True)

    num_total_points = len(xyz)

    if num_total_points > MAX_RETURN_POINTS:
        idx = np.random.choice(num_total_points, MAX_RETURN_POINTS, replace=False)
        xyz_vis = xyz[idx]
        semantic_ids_vis = semantic_ids[idx]
        instance_ids_vis = instance_ids[idx]
        semantic_scores_vis = semantic_scores[idx]
    else:
        xyz_vis = xyz
        semantic_ids_vis = semantic_ids
        instance_ids_vis = instance_ids
        semantic_scores_vis = semantic_scores

    print("[segment] building response payload", flush=True)

    points_out = []
    for i in range(len(xyz_vis)):
        points_out.append([
            float(xyz_vis[i, 0]),
            float(xyz_vis[i, 1]),
            float(xyz_vis[i, 2]),
            int(instance_ids_vis[i]),
            int(semantic_ids_vis[i]),
            float(np.max(semantic_scores_vis[i])),
        ])

    return {
        "num_points": int(num_total_points),
        "num_points_returned": int(len(points_out)),
        "num_instances": int(len(instances)),
        "instances": instances,
        "points": points_out,
    }