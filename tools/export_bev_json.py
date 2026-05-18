"""Export SparseDrive BEV predictions from a results.pkl to a portable JSON.

Usage:
    python tools/export_bev_json.py \\
        work_dirs/sparsedrive_small_stage2/results_trt.pkl \\
        work_dirs/sparsedrive_small_stage2/bev_predictions.json \\
        --det_score_thr 0.25 --map_score_thr 0.4
"""

import argparse
import json
import math

import mmcv
import numpy as np
import torch


DET_CLASS_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
MAP_CLASS_NAMES = ["divider", "ped_crossing", "boundary"]


def t2n(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def decode_box(b10):
    """SparseDrive box: (x, y, z, log_w, log_l, log_h, sin_yaw, cos_yaw, vx, vy)."""
    x, y, z, lw, ll, lh, sy, cy, vx, vy = [float(v) for v in b10]
    return {
        "x": x, "y": y, "z": z,
        "w": math.exp(lw), "l": math.exp(ll), "h": math.exp(lh),
        "yaw": math.atan2(sy, cy),
        "vx": vx, "vy": vy,
    }


def export(pkl_path, out_path, det_thr, map_thr, traj_topk):
    results = mmcv.load(pkl_path)
    frames = []
    for idx, r in enumerate(results):
        d = r["img_bbox"]

        boxes = t2n(d["boxes_3d"])           # (300, 10)
        scores = t2n(d["scores_3d"])         # (300,)
        labels = t2n(d["labels_3d"]).astype(int)
        ids = t2n(d["instance_ids"]).astype(int)
        trajs = t2n(d["trajs_3d"])           # (300, 6, 12, 2)
        traj_scores = t2n(d["trajs_score"])  # (300, 6)

        det = []
        for i in np.where(scores >= det_thr)[0]:
            box = decode_box(boxes[i])
            # Pick top-k motion modes for this object
            order = np.argsort(-traj_scores[i])[:traj_topk]
            modes = [
                {
                    "score": float(traj_scores[i, m]),
                    "xy": trajs[i, m].tolist(),     # (12, 2)
                }
                for m in order
            ]
            det.append({
                "id": int(ids[i]),
                "class": DET_CLASS_NAMES[int(labels[i])] if int(labels[i]) < len(DET_CLASS_NAMES) else str(int(labels[i])),
                "class_id": int(labels[i]),
                "score": float(scores[i]),
                **box,
                "motion_modes": modes,
            })

        # Map vectors: list of 100; each is (20, 2)
        vectors = d["vectors"]
        m_scores = t2n(d["scores"])
        m_labels = t2n(d["labels"]).astype(int)
        map_elems = []
        for i, v in enumerate(vectors):
            if m_scores[i] < map_thr:
                continue
            map_elems.append({
                "class": MAP_CLASS_NAMES[int(m_labels[i])] if int(m_labels[i]) < len(MAP_CLASS_NAMES) else str(int(m_labels[i])),
                "class_id": int(m_labels[i]),
                "score": float(m_scores[i]),
                "polyline": t2n(v).tolist(),
            })

        # Planning: (3 commands, 6 modes, 6 steps, 2). final_planning: (6, 2)
        plan_scores = t2n(d["planning_score"])      # (3, 6)
        plan_xy = t2n(d["planning"])                # (3, 6, 6, 2)
        final = t2n(d["final_planning"]).tolist()   # (6, 2)
        plan_modes = []
        for c in range(plan_scores.shape[0]):
            for m in range(plan_scores.shape[1]):
                plan_modes.append({
                    "command": c, "mode": m,
                    "score": float(plan_scores[c, m]),
                    "xy": plan_xy[c, m].tolist(),
                })
        plan_modes.sort(key=lambda e: -e["score"])

        frames.append({
            "frame_idx": idx,
            "detections": det,
            "map_elements": map_elems,
            "planning": {
                "final_xy": final,           # ego future trajectory in ego frame
                "modes_topk": plan_modes[:6],
            },
        })

    meta = {
        "format_version": 1,
        "frame_count": len(frames),
        "frame_rate_hz": 2,
        "frame_convention": "ego_lidar",
        "axes": "x forward (+), y left (+), z up (+); meters",
        "det_score_threshold": det_thr,
        "map_score_threshold": map_thr,
        "traj_horizon_steps": 12,
        "traj_step_seconds": 0.5,
        "plan_horizon_steps": 6,
        "plan_step_seconds": 0.5,
        "det_class_names": DET_CLASS_NAMES,
        "map_class_names": MAP_CLASS_NAMES,
        "notes": [
            "All coordinates are in the ego/LiDAR frame at the current frame's timestamp.",
            "Motion trajectories (per-object) are 6s into the future at 2Hz (12 steps).",
            "Ego planning is 3s into the future at 2Hz (6 steps).",
        ],
    }

    with open(out_path, "w") as f:
        json.dump({"meta": meta, "frames": frames}, f)
    print(f"Wrote {out_path} ({len(frames)} frames)")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", help="results pkl from test_trt.py")
    ap.add_argument("out", help="output JSON path")
    ap.add_argument("--det_score_thr", type=float, default=0.25)
    ap.add_argument("--map_score_thr", type=float, default=0.4)
    ap.add_argument("--traj_topk", type=int, default=3)
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    export(a.pkl, a.out, a.det_score_thr, a.map_score_thr, a.traj_topk)
