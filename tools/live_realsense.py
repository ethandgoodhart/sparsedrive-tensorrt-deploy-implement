#!/usr/bin/env python
"""Live BEV visualization from a RealSense color stream.

Reads color frames from a RealSense camera, replicates the single view across
all 6 expected camera inputs (with projection matrices borrowed from the
nuScenes mini sample so the geometry is at least plausible), runs the
perception + motion engine cascade, and displays a side-by-side camera/BEV
view with an FPS overlay.

This is a visualization demo — feeding the same view to all 6 cameras is
deliberately wrong for 3D perception, so detected boxes are approximate. The
purpose is to show the live engine pipeline running at cart-deployment rate.
"""
import argparse
import ctypes
import os
import sys
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import cv2
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import projects.mmdet3d_plugin  # noqa: F401
from tools.test_trt import TRTInfer

# Model input shape: 6 cameras x 3 channels x 256 x 704, ImageNet normalized, RGB
IMG_H, IMG_W = 256, 704
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD  = np.array([ 58.395,  57.12,  57.375], dtype=np.float32)

# BEV canvas: 60m x 60m at 0.1 m/px → 600 x 600 px, ego at center.
BEV_PX  = 600
BEV_RES = 0.1  # m / px
EGO_PX  = (BEV_PX // 2, BEV_PX // 2)

CLASS_COLORS = [
    (255, 64, 64), (255, 128, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255),
    (0, 128, 255), (128, 0, 255), (255, 0, 255), (200, 200, 200), (255, 255, 255),
]


def grab_projection_mat(cfg_path):
    """Pull one set of 6 projection matrices from the nuScenes mini sample.

    Used so the model sees plausible camera intrinsics even though we replicate
    the same RealSense image to all 6 slots.
    """
    from mmcv import Config
    from mmcv.parallel.scatter_gather import scatter
    from mmdet.datasets import build_dataset
    from mmdet.datasets import build_dataloader as build_dataloader_origin
    cfg = Config.fromfile(cfg_path)
    ds = build_dataset(cfg.data.test)
    loader = build_dataloader_origin(ds, samples_per_gpu=1, workers_per_gpu=1,
                                     dist=False, shuffle=False)
    for d in loader:
        s = scatter(d, [torch.cuda.current_device()])[0]
        return s['projection_mat'].contiguous(), s['img_metas'][0]


def preprocess_realsense(bgr, dst_h=IMG_H, dst_w=IMG_W):
    """RealSense BGR uint8 frame → 6×3×H×W float32 GPU tensor."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
    rgb = rgb.astype(np.float32)
    rgb = (rgb - MEAN) / STD
    rgb = np.transpose(rgb, (2, 0, 1))                # 3,H,W
    six = np.broadcast_to(rgb, (6,) + rgb.shape).copy()  # 6,3,H,W
    return torch.from_numpy(six).unsqueeze(0).cuda()  # 1,6,3,H,W


def decode_boxes(det_bbox, det_cls, top_k=20, score_thr=0.25):
    """Decode SparseDrive 11-dim anchors → list of (x, y, w, l, yaw, score, cls).

    Layout: x, y, z, log_w, log_l, log_h, sin_yaw, cos_yaw, vx, vy, vz.
    Returns boxes in the ego frame (x forward, y left, BEV plane).
    """
    bbox = det_bbox[0].float().cpu().numpy()   # 900, 11
    cls  = det_cls[0].float().cpu().numpy()    # 900, num_cls
    # Sigmoid → top class per anchor
    p = 1.0 / (1.0 + np.exp(-cls))
    cls_idx = p.argmax(axis=1)
    score   = p.max(axis=1)
    keep = score > score_thr
    if not np.any(keep):
        return []
    bbox, score, cls_idx = bbox[keep], score[keep], cls_idx[keep]
    order = np.argsort(-score)[:top_k]
    bbox, score, cls_idx = bbox[order], score[order], cls_idx[order]
    out = []
    for b, s, c in zip(bbox, score, cls_idx):
        x, y, z = b[0], b[1], b[2]
        w, l    = float(np.exp(b[3])), float(np.exp(b[4]))
        yaw     = float(np.arctan2(b[6], b[7]))
        out.append((float(x), float(y), w, l, yaw, float(s), int(c)))
    return out


def world_to_bev(x, y):
    """Ego frame (x=forward, y=left) → BEV pixel coords (BEV up = +x)."""
    px = int(EGO_PX[0] - y / BEV_RES)
    py = int(EGO_PX[1] - x / BEV_RES)
    return px, py


def draw_box_bev(canvas, box):
    x, y, w, l, yaw, score, cls = box
    # Rectangle corners in box-local frame
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    hl, hw = l / 2.0, w / 2.0
    corners = np.array([[ hl,  hw], [ hl, -hw], [-hl, -hw], [-hl,  hw]])
    corners = corners @ R.T + np.array([x, y])
    pts = np.array([world_to_bev(cx, cy) for cx, cy in corners], dtype=np.int32)
    color = CLASS_COLORS[cls % len(CLASS_COLORS)]
    cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
    # forward heading marker
    hx, hy = x + np.cos(yaw) * (l / 2.0), y + np.sin(yaw) * (l / 2.0)
    cv2.line(canvas, world_to_bev(x, y), world_to_bev(hx, hy), color, 2)


def draw_trajectory(canvas, traj_xy, color=(0, 255, 0), thickness=3):
    pts = np.array([world_to_bev(p[0], p[1]) for p in traj_xy], dtype=np.int32)
    if len(pts) >= 2:
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness)
    for px, py in pts:
        cv2.circle(canvas, (px, py), 3, color, -1)


def draw_map_polylines(canvas, map_pts, map_cls, score_thr=0.4):
    """map_pts: (100, 40) flattened polylines (20 pts × 2). map_cls: (100, 3)."""
    pts = map_pts[0].float().cpu().numpy().reshape(-1, 20, 2)  # 100, 20, 2
    cls = map_cls[0].float().cpu().numpy()
    p = 1.0 / (1.0 + np.exp(-cls))
    cls_idx = p.argmax(axis=1)
    score   = p.max(axis=1)
    map_colors = [(80, 255, 80), (80, 80, 255), (255, 200, 80)]  # ped, divider, boundary
    for i in range(pts.shape[0]):
        if score[i] < score_thr:
            continue
        poly = np.array([world_to_bev(p[0], p[1]) for p in pts[i]], dtype=np.int32)
        cv2.polylines(canvas, [poly], False, map_colors[cls_idx[i] % 3], 1)


def render_bev(det_bbox, det_cls, plan_reg, plan_cls, map_pts, map_cls):
    canvas = np.full((BEV_PX, BEV_PX, 3), 24, dtype=np.uint8)  # dark gray
    # Grid every 10m
    for r in range(10, 31, 10):
        cv2.circle(canvas, EGO_PX, int(r / BEV_RES), (60, 60, 60), 1)
    # Ego (5m × 2m car)
    ego_corners = [(2.5, 1.0), (2.5, -1.0), (-2.5, -1.0), (-2.5, 1.0)]
    ego_px = np.array([world_to_bev(x, y) for x, y in ego_corners], dtype=np.int32)
    cv2.fillPoly(canvas, [ego_px], (0, 200, 255))

    draw_map_polylines(canvas, map_pts, map_cls)

    for box in decode_boxes(det_bbox, det_cls):
        draw_box_bev(canvas, box)

    # Top-1 planning trajectory (across all 18 modes)
    pc = plan_cls[0, 0].float().cpu().numpy()           # 18
    pr = plan_reg[0, 0].float().cpu().numpy()           # 18, 6, 2
    top = int(pc.argmax())
    draw_trajectory(canvas, pr[top])

    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="config (for borrowing projection_mat)")
    ap.add_argument("--engine_perc_init", default="work_dirs/sparsedrive_small_stage2/sparsedrive_multihead_first.engine")
    ap.add_argument("--engine_perc_temp", default="work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.engine")
    ap.add_argument("--engine_mo_init",   default="work_dirs/sparsedrive_small_stage2/motion_plan_engine_first.engine")
    ap.add_argument("--engine_mo_temp",   default="work_dirs/sparsedrive_small_stage2/motion_plan_engine.engine")
    ap.add_argument("--plugin",           default="projects/trt_plugin/build/libSparseDrivePlugin.so")
    ap.add_argument("--rs_width",  type=int, default=640)
    ap.add_argument("--rs_height", type=int, default=480)
    ap.add_argument("--rs_fps",    type=int, default=30)
    ap.add_argument("--display",   default="auto", choices=["auto", "window", "file"])
    ap.add_argument("--out",       default="work_dirs/sparsedrive_small_stage2/live.mp4")
    ap.add_argument("--max_seconds", type=float, default=0.0,
                    help="auto-quit after N seconds (0 = run until Ctrl-C or 'q')")
    args = ap.parse_args()

    print("🔧 Loading plugin + engines ...")
    ctypes.CDLL(args.plugin, mode=ctypes.RTLD_GLOBAL)
    perc_init = TRTInfer(args.engine_perc_init)
    perc_temp = TRTInfer(args.engine_perc_temp)
    mo_init   = TRTInfer(args.engine_mo_init)
    mo_temp   = TRTInfer(args.engine_mo_temp)

    nh_det  = perc_temp.inputs['prev_det_feat'].shape[1]
    dim_det = perc_temp.inputs['prev_det_anchor'].shape[2]
    nh_map  = perc_temp.inputs['prev_map_feat'].shape[1]
    dim_map = perc_temp.inputs['prev_map_anchor'].shape[2]
    Q       = mo_temp.inputs['mo_history_anchor'].shape[2]
    print(f"   sizes: nh_det={nh_det} nh_map={nh_map} Q={Q}")

    print("🗺️  Borrowing projection matrices from nuScenes mini frame 0 ...")
    proj_mat, _ = grab_projection_mat(args.config)

    print("🎥 Starting RealSense color stream ...")
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.rs_width, args.rs_height,
                         rs.format.bgr8, args.rs_fps)
    profile = pipeline.start(config)

    def zero_hist_det():
        return {
            'prev_det_feat':   torch.zeros((1, nh_det, 256), device='cuda'),
            'prev_det_anchor': torch.zeros((1, nh_det, dim_det), device='cuda'),
            'prev_det_conf':   torch.zeros((1, nh_det), device='cuda'),
            'prev_det_id':     torch.full((1, nh_det), -1, dtype=torch.int32, device='cuda'),
            'prev_id_count':   torch.zeros((1, 1), dtype=torch.int32, device='cuda'),
        }
    def zero_hist_map():
        return {
            'prev_map_feat':   torch.zeros((1, nh_map, 256), device='cuda'),
            'prev_map_anchor': torch.zeros((1, nh_map, dim_map), device='cuda'),
            'prev_map_conf':   torch.zeros((1, nh_map), device='cuda'),
        }
    def zero_hist_motion():
        return {
            "mo_history_instance_feature": torch.zeros((1, nh_det, Q, 256), device='cuda'),
            "mo_history_anchor":           torch.zeros((1, nh_det, Q, 11), device='cuda'),
            "mo_history_period":           torch.zeros((1, nh_det), dtype=torch.int32, device='cuda'),
            "mo_prev_instance_id":         torch.zeros((1, nh_det), dtype=torch.int32, device='cuda'),
            "mo_prev_confidence":          torch.zeros((1, nh_det), device='cuda'),
            "mo_history_ego_feature":      torch.zeros((1, 1, Q, 256), device='cuda'),
            "mo_history_ego_anchor":       torch.zeros((1, 1, Q, 11), device='cuda'),
            "mo_history_ego_period":       torch.zeros((1, 1), dtype=torch.int32, device='cuda'),
            "mo_prev_ego_status":          torch.zeros((1, 1, 10), device='cuda'),
        }

    hist_d  = zero_hist_det()
    hist_m  = zero_hist_map()
    hist_mo = zero_hist_motion()
    first   = True
    identity_t = torch.eye(4, device='cuda').unsqueeze(0)

    # Display target
    display_mode = args.display
    if display_mode == "auto":
        display_mode = "window" if os.environ.get("DISPLAY") else "file"
    print(f"🖥️  Display mode: {display_mode}")
    writer = None
    if display_mode == "window":
        cv2.namedWindow("SparseDrive Live (camera | BEV)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SparseDrive Live (camera | BEV)", 1200, 600)

    # Warmup: 3 frames so engine timing is steady
    print("🔥 Warming up ...")
    for _ in range(3):
        f = pipeline.wait_for_frames()
        bgr = np.asanyarray(f.get_color_frame().get_data())
        _ = preprocess_realsense(bgr)

    fps_ema = 0.0
    alpha   = 0.2
    last_t  = time.perf_counter()
    t_start = last_t

    print("▶️  Running. Press 'q' in the window to quit (or Ctrl-C).")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color  = frames.get_color_frame()
            if not color:
                continue
            bgr = np.asanyarray(color.get_data())

            t0 = time.perf_counter()
            img_tensor = preprocess_realsense(bgr)
            dt_tensor   = torch.tensor([0.5], device='cuda', dtype=torch.float32)
            mask_tensor = torch.tensor([not first], device='cuda', dtype=torch.bool)

            feed_perc = {
                'img': img_tensor, 'projection_mat': proj_mat,
                'instance_t_matrix': identity_t, 'time_interval': dt_tensor,
                **hist_d, **hist_m,
            }
            out_p = (perc_init if first else perc_temp).infer(feed_perc)

            hist_d = {
                'prev_det_feat':   out_p['next_det_feat'],
                'prev_det_anchor': out_p['next_det_anchor'],
                'prev_det_conf':   out_p['next_det_conf'],
                'prev_det_id':     out_p['next_det_instance_id'],
                'prev_id_count':   out_p['next_id_count'],
            }
            hist_m = {
                'prev_map_feat':   out_p['next_map_feat'],
                'prev_map_anchor': out_p['next_map_anchor'],
                'prev_map_conf':   out_p['next_map_conf'],
            }

            feed_mo = {
                'det_instance_feature':       out_p['det_instance_feature'],
                'det_anchor_embed':           out_p['det_anchor_embed'],
                'det_classification':         out_p['det_cls'],
                'det_anchors':                out_p['det_bbox'],
                'det_instance_id':            out_p['det_instance_id'].to(torch.int32),
                'map_instance_feature':       out_p['map_instance_feature'],
                'map_anchor_embed':           out_p['map_anchor_embed'],
                'map_classification':         out_p['map_cls'],
                'ego_feature_map':            out_p['ego_feature_map'],
                'instance_t_matrix':          identity_t,
                'mask':                       mask_tensor,
                **hist_mo,
            }
            out_m = (mo_init if first else mo_temp).infer(feed_mo)

            hist_mo = {
                'mo_history_instance_feature': out_m['next_mo_history_instance_feature'],
                'mo_history_anchor':           out_m['next_mo_history_anchor'],
                'mo_history_period':           out_m['next_mo_history_period'],
                'mo_prev_instance_id':         out_m['next_mo_prev_instance_id'],
                'mo_prev_confidence':          out_m['next_mo_prev_confidence'],
                'mo_history_ego_feature':      out_m['next_mo_history_ego_feature'],
                'mo_history_ego_anchor':       out_m['next_mo_history_ego_anchor'],
                'mo_history_ego_period':       out_m['next_mo_history_ego_period'],
                'mo_prev_ego_status':          out_m['next_mo_prev_ego_status'],
            }
            torch.cuda.synchronize()
            t_infer = (time.perf_counter() - t0) * 1000

            first = False

            bev = render_bev(
                out_p['det_bbox'], out_p['det_cls'],
                out_m['plan_reg'], out_m['plan_cls'],
                out_p['map_pts'], out_p['map_cls'],
            )

            # Camera panel: resize to match BEV height
            cam_h = BEV_PX
            cam_w = int(bgr.shape[1] * cam_h / bgr.shape[0])
            cam_panel = cv2.resize(bgr, (cam_w, cam_h))
            frame = np.hstack([cam_panel, bev])

            # FPS overlay
            now = time.perf_counter()
            inst_fps = 1.0 / max(now - last_t, 1e-6)
            fps_ema = inst_fps if fps_ema == 0 else (1 - alpha) * fps_ema + alpha * inst_fps
            last_t = now
            txt = f"{fps_ema:5.1f} FPS  |  infer {t_infer:5.1f} ms"
            cv2.putText(frame, txt, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, txt, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 255), 2, cv2.LINE_AA)

            if display_mode == "window":
                cv2.imshow("SparseDrive Live (camera | BEV)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
            else:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'),
                                              20.0, (w, h))
                writer.write(frame)

            if args.max_seconds > 0 and (now - t_start) > args.max_seconds:
                break
    finally:
        pipeline.stop()
        if writer is not None:
            writer.release()
            print(f"💾 Saved → {args.out}")
        if display_mode == "window":
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
