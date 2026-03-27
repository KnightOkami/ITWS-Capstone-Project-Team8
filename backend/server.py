import json
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from models import pointnet_cls
import os
import shutil
from worker import meta_path, start_worker, RAW_DIR, POINTCLOUD_DIR, write_meta, Job, job_queue
from segmenter import load_xyz_from_txt, sample_points, cluster_pointcloud
from ml_segmenter import run_ml_segmentation
from pathlib import Path

SEGMENT_CLASSES = [
    "ceiling", "floor", "wall", "beam", "column", "window", "door",
    "table", "chair", "sofa", "bookcase", "board", "clutter"
]

OFFICIAL_VISUAL_DIR = Path("/app/third_party/Pointnet_Pointnet2_pytorch/log/sem_seg/pointnet2_sem_seg/visual")
MAX_OFFICIAL_RETURN_POINTS = 200000

app = Flask(__name__)
CORS(app, supports_credentials=True)

# === Load model ===
MODEL_PATH = 'log/model.ckpt'
NUM_POINTS = 2048
NUM_CLASSES = 40

pointclouds_pl = tf.placeholder(tf.float32, shape=(1, NUM_POINTS, 3))
is_training_pl = tf.placeholder(tf.bool, shape=())
pred, _ = pointnet_cls.get_model(pointclouds_pl, is_training_pl)
pred_softmax = tf.nn.softmax(pred)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, MODEL_PATH)

classes = [
    'airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone',
    'cup','curtain','desk','door','dresser','flower_pot','glass_box','guitar','keyboard',
    'lamp','laptop','mantel','monitor','night_stand','person','piano','plant','radio',
    'range_hood','sink','sofa','stairs','stool','table','tent','toilet','tv_stand',
    'vase','wardrobe','xbox'
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    try:
        points = np.loadtxt(file).astype(np.float32)
    except Exception as e:
        return jsonify({"error": f"Failed to parse point cloud file: {str(e)}"}), 400

    # Handle edge case: single row becomes 1D
    if points.ndim == 1:
        if points.shape[0] < 3:
            return jsonify({"error": "Point cloud must contain at least 3 values per row (X Y Z)"}), 400
        points = np.expand_dims(points, axis=0)

    # Must be a 2D array with at least 3 columns
    if points.ndim != 2 or points.shape[1] < 3:
        return jsonify({"error": "Point cloud file must have at least 3 columns (X Y Z)"}), 400

    # Ignore any extra columns such as RGB, normals, intensity, labels, etc.
    points = points[:, :3]

    # pad or sample to 2048
    if points.shape[0] > NUM_POINTS:
        idx = np.random.choice(points.shape[0], NUM_POINTS, replace=False)
        points = points[idx, :]
    elif points.shape[0] < NUM_POINTS:
        pad = np.zeros((NUM_POINTS - points.shape[0], 3), dtype=np.float32)
        points = np.vstack((points, pad))

    points = np.expand_dims(points, axis=0)

    pred_val = sess.run(
        pred_softmax,
        feed_dict={pointclouds_pl: points, is_training_pl: False}
    )
    pred_class = np.argmax(pred_val, axis=1)[0]
    confidence = float(np.max(pred_val))

    return jsonify({
        "predicted_index": int(pred_class),
        "predicted_label": classes[pred_class],
        "confidence": round(confidence * 100, 2)
    })

@app.route('/segment', methods=['POST'])
def segment():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    try:
        result = run_ml_segmentation(file)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"ML segmentation failed: {str(e)}"}), 400
    

@app.route('/potree/libs/<path:filename>')
def potree_libs(filename):
    potree_libs_dir = os.path.join(os.path.dirname(__file__), 'potree', 'libs')
    return send_from_directory(potree_libs_dir, filename)

@app.route('/pointclouds/<cloud_name>/<path:filename>')
def pointcloud_file(cloud_name, filename):
    base_dir = os.path.join(os.path.dirname(__file__), 'data', 'pointclouds', cloud_name)
    return send_from_directory(base_dir, filename)

@app.route('/viewer/<cloud_name>')
def viewer(cloud_name):
    pointcloud_url = f'/pointclouds/{cloud_name}/metadata.json'
    return render_template('viewer.html', pointcloud_url=pointcloud_url, cloud_name=cloud_name)

@app.route("/api/pointclouds", methods=["GET"])
def list_pointclouds():
    status_map = {}

    for name in os.listdir(POINTCLOUD_DIR):
        mp = meta_path(name)
        if os.path.exists(mp):
            with open(mp) as f:
                meta = json.load(f)
                status = meta.get("status", "unknown")
                error = meta.get("error", None)
                desc = meta.get("description", "")
                status_map.setdefault(status, []).append((name, error, desc))
    return jsonify(status_map)


@app.route("/api/pointclouds/upload", methods=["POST"])
def upload_pointcloud():
    file = request.files.get("file")
    name = request.form.get("name")
    description = request.form.get("description")

    if not file or not name:
        return jsonify({"error": "Missing file or name"}), 400

    pc_dir = os.path.join(POINTCLOUD_DIR, name)
    if os.path.exists(pc_dir):
        return jsonify({"error": "A pointcloud with this name already exists"}), 400

    extension = file.filename.split('.')[-1].lower()

    if extension != 'laz' and extension != 'las':
        return jsonify({"error": "Only .laz and .las files are supported"}), 400

    # Save raw file
    raw_path = os.path.join(RAW_DIR, f"{name}.{extension}")
    file.save(raw_path)

    # Create folder
    os.makedirs(pc_dir)

    # Create meta.json
    meta = {
        "name": name,
        "description": description,
        "status": "pending"
    }
    write_meta(name, meta)

    # Queue job
    job_queue.put(Job(name))

    return jsonify({"message": "Upload successful", "name": name})

@app.route('/api/pointclouds/delete', methods=['POST'])
def delete_pointcloud():
    data = request.get_json()
    name = data.get('name')

    if not name:
        return jsonify({"error": "Missing pointcloud name"}), 400

    pc_dir = os.path.join(POINTCLOUD_DIR, name)
    raw_path = os.path.join(RAW_DIR, f"{name}.laz")
    raw_path_2 = os.path.join(RAW_DIR, f"{name}.las")

    # Remove the pointcloud directory if it exists
    if os.path.exists(pc_dir) and os.path.isdir(pc_dir):
        shutil.rmtree(pc_dir)

    # Remove the raw file if it exists
    if os.path.exists(raw_path) and os.path.isfile(raw_path):
        os.remove(raw_path)

    if os.path.exists(raw_path_2) and os.path.isfile(raw_path_2):
        os.remove(raw_path_2)

    return jsonify({"message": "Pointcloud deleted", "name": name})

@app.route("/debug/jobqueue")
def debug_jobqueue():
    q = list(job_queue.queue)
    return jsonify([job.name for job in q])

@app.route('/official_segment_files', methods=['GET'])
def official_segment_files():
    if not OFFICIAL_VISUAL_DIR.exists():
        return jsonify({"error": "Official segmentation output directory not found"}), 404

    files = sorted([p.name for p in OFFICIAL_VISUAL_DIR.glob("*_pred.obj")])
    return jsonify({"files": files})


@app.route('/official_segment_result/<path:filename>', methods=['GET'])
def official_segment_result(filename):
    file_path = OFFICIAL_VISUAL_DIR / filename

    if not file_path.exists() or not file_path.is_file():
        return jsonify({"error": "Segmentation result file not found"}), 404

    # If user selected the labels-only .txt, redirect internally to the matching _pred.obj
    if file_path.suffix == ".txt":
        stem = file_path.stem
        obj_candidate = OFFICIAL_VISUAL_DIR / f"{stem}_pred.obj"
        if obj_candidate.exists():
            file_path = obj_candidate
        else:
            return jsonify({"error": "Could not find matching _pred.obj file"}), 404

    # Only support OBJ here
    if file_path.suffix.lower() != ".obj":
        return jsonify({"error": "Expected an OBJ file"}), 400

    try:
        xyz_list = []
        rgb_list = []

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("v "):
                    continue

                parts = line.split()
                # Expected: v x y z r g b
                if len(parts) < 7:
                    continue

                x, y, z = map(float, parts[1:4])
                r, g, b = map(int, parts[4:7])

                xyz_list.append([x, y, z])
                rgb_list.append([r, g, b])

        if not xyz_list:
            return jsonify({"error": "No vertex data found in OBJ"}), 400

        xyz_raw = np.array(xyz_list, dtype=np.float32)
        rgb = np.array(rgb_list, dtype=np.int32)

        # Map RGB back to semantic label using the standard S3DIS color map
        color_to_label = {
            (0, 255, 0): 0,        # ceiling
            (0, 0, 255): 1,        # floor
            (0, 255, 255): 2,      # wall
            (255, 255, 0): 3,      # beam
            (255, 0, 255): 4,      # column
            (100, 100, 255): 5,    # window
            (200, 200, 100): 6,    # door
            (170, 120, 200): 7,    # table
            (255, 0, 0): 8,        # chair
            (200, 100, 100): 9,    # sofa
            (10, 200, 100): 10,    # bookcase
            (200, 200, 200): 11,   # board
            (50, 50, 50): 12,      # clutter
        }

        semantic_ids = np.array(
            [color_to_label.get(tuple(c), 12) for c in rgb],
            dtype=np.int32
        )

        # Center + scale for Plotly viewer
        center = np.mean(xyz_raw, axis=0, keepdims=True)
        xyz = xyz_raw - center

        scale = np.max(np.linalg.norm(xyz, axis=1))
        if scale > 0:
            xyz = xyz / scale

        num_total_points = len(xyz)

        if num_total_points > MAX_OFFICIAL_RETURN_POINTS:
            idx = np.random.choice(num_total_points, MAX_OFFICIAL_RETURN_POINTS, replace=False)
            xyz = xyz[idx]
            semantic_ids = semantic_ids[idx]

        points_out = []
        for i in range(len(xyz)):
            label = int(semantic_ids[i])
            points_out.append([
                float(xyz[i, 0]),
                float(xyz[i, 1]),
                float(xyz[i, 2]),
                label,   # use semantic label as color/group key
                label,
                1.0
            ])

        instances = []
        for label in sorted(set(semantic_ids.tolist())):
            class_xyz = xyz[semantic_ids == label]
            if len(class_xyz) == 0:
                continue

            class_name = SEGMENT_CLASSES[label] if 0 <= label < len(SEGMENT_CLASSES) else f"class_{label}"

            instances.append({
                "id": int(label),
                "semantic_id": int(label),
                "semantic_label": class_name,
                "count": int(len(class_xyz)),
                "bbox_min": [float(v) for v in class_xyz.min(axis=0).tolist()],
                "bbox_max": [float(v) for v in class_xyz.max(axis=0).tolist()],
            })

        return jsonify({
            "file_name": file_path.name,
            "num_points": int(num_total_points),
            "num_points_returned": int(len(points_out)),
            "num_instances": int(len(instances)),
            "instances": instances,
            "points": points_out,
        })

    except Exception as e:
        return jsonify({"error": f"Failed to load official segmentation file: {str(e)}"}), 400
start_worker()
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(POINTCLOUD_DIR, exist_ok=True) 

if __name__ == '__main__':
    # Enable debug mode based on environment
    debug_mode = os.getenv('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug_mode, host='0.0.0.0', port=8080)
