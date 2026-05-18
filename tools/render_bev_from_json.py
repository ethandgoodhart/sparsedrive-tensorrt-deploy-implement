"""Render a BEV video from bev_predictions.json (no model needed).

Usage:
    python tools/render_bev_from_json.py \\
        work_dirs/sparsedrive_small_stage2/bev_predictions.json \\
        work_dirs/sparsedrive_small_stage2/bev_from_json.mp4
"""

import argparse
import json
import math

import cv2
import numpy as np


CANVAS = 800
RES = 0.1               # meters per pixel
EGO_PX = (CANVAS // 2, CANVAS // 2)
FPS = 2

DET_COLORS = [
    (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
    (250, 190, 212), (0, 128, 128),
]
MAP_COLORS = [(255, 255, 255), (0, 255, 255), (255, 0, 255)]


def w2p(x, y):
    """Ego (x fwd, y left) -> BEV pixel (image coords). Forward = up."""
    px = int(EGO_PX[0] - y / RES)
    py = int(EGO_PX[1] - x / RES)
    return px, py


def draw_grid(canvas):
    for r_m in range(10, 50, 10):
        cv2.circle(canvas, EGO_PX, int(r_m / RES), (40, 40, 40), 1)
    cv2.line(canvas, (EGO_PX[0], 0), (EGO_PX[0], CANVAS), (40, 40, 40), 1)
    cv2.line(canvas, (0, EGO_PX[1]), (CANVAS, EGO_PX[1]), (40, 40, 40), 1)


def draw_ego(canvas):
    pts = np.array([w2p(2.0, 0.9), w2p(2.0, -0.9), w2p(-2.0, -0.9), w2p(-2.0, 0.9)], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], (200, 200, 200))


def draw_box(canvas, det):
    x, y = det["x"], det["y"]
    w, l, yaw = det["w"], det["l"], det["yaw"]
    if not all(map(math.isfinite, (x, y, w, l, yaw))):
        return
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = l / 2.0, w / 2.0
    corners = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    pts = np.array([w2p(x + cx * c - cy * s, y + cx * s + cy * c) for cx, cy in corners], dtype=np.int32)
    color = DET_COLORS[det["class_id"] % len(DET_COLORS)]
    cv2.polylines(canvas, [pts], True, color, 2)
    hx, hy = x + math.cos(yaw) * hl, y + math.sin(yaw) * hl
    cv2.line(canvas, w2p(x, y), w2p(hx, hy), color, 2)


def draw_polyline(canvas, xy, color, thickness=2, closed=False):
    pts = []
    for p in xy:
        if not (math.isfinite(p[0]) and math.isfinite(p[1])):
            continue
        pts.append(w2p(p[0], p[1]))
    if len(pts) >= 2:
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], closed, color, thickness)


def render_frame(frame, meta):
    canvas = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
    draw_grid(canvas)
    for elem in frame["map_elements"]:
        color = MAP_COLORS[elem["class_id"] % len(MAP_COLORS)]
        draw_polyline(canvas, elem["polyline"], color, 1)
    for det in frame["detections"]:
        draw_box(canvas, det)
        if det["motion_modes"]:
            draw_polyline(canvas, det["motion_modes"][0]["xy"], (0, 200, 255), 1)
    if frame["planning"]["modes_topk"]:
        draw_polyline(canvas, frame["planning"]["modes_topk"][0]["xy"], (0, 255, 0), 3)
    draw_polyline(canvas, frame["planning"]["final_xy"], (0, 255, 0), 2)
    draw_ego(canvas)
    cv2.putText(canvas, f"frame {frame['frame_idx']:03d}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("out_mp4")
    ap.add_argument("--fps", type=int, default=FPS)
    args = ap.parse_args()

    with open(args.json_path) as f:
        data = json.load(f)
    meta, frames = data["meta"], data["frames"]

    writer = cv2.VideoWriter(
        args.out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (CANVAS, CANVAS)
    )
    for fr in frames:
        writer.write(render_frame(fr, meta))
    writer.release()
    print(f"Wrote {args.out_mp4} ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
