#!/usr/bin/env python
"""Live-cart inference simulation.

Streams the nuScenes mini clip frames through the perception + motion engines at
a fixed simulated camera rate. Measures capture→output latency and inference
throughput. All frames pre-loaded to RAM to simulate that camera capture and
preprocessing run on a separate thread/process (which is what a real cart node
would do).
"""
import argparse
import ctypes
import os
import sys
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmcv import Config
from mmcv.parallel.scatter_gather import scatter
from mmdet.datasets import build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin

import projects.mmdet3d_plugin  # noqa: F401
from tools.test_trt import TRTInfer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--engine_perc_init", default="work_dirs/sparsedrive_small_stage2/sparsedrive_multihead_first.engine")
    ap.add_argument("--engine_perc_temp", default="work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.engine")
    ap.add_argument("--engine_mo_init",   default="work_dirs/sparsedrive_small_stage2/motion_plan_engine_first.engine")
    ap.add_argument("--engine_mo_temp",   default="work_dirs/sparsedrive_small_stage2/motion_plan_engine.engine")
    ap.add_argument("--plugin",           default="projects/trt_plugin/build/libSparseDrivePlugin.so")
    ap.add_argument("--rate_hz", type=float, default=10.0,
                    help="simulated camera capture rate; 0 = unbounded (max FPS)")
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    ctypes.CDLL(args.plugin, mode=ctypes.RTLD_GLOBAL)
    cfg = Config.fromfile(args.config)

    # Build dataset/dataloader and preload all formatted frames into GPU memory
    print("📦 Loading and preprocessing all frames ...")
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader_origin(dataset, samples_per_gpu=1, workers_per_gpu=2,
                                     dist=False, shuffle=False)
    frames = []
    for data in loader:
        s = scatter(data, [torch.cuda.current_device()])[0]
        frames.append({
            'img': s['img'].contiguous(),
            'projection_mat': s['projection_mat'].contiguous(),
            'timestamp': s['img_metas'][0]['timestamp'],
            'T_global': s['img_metas'][0]['T_global'],
            'T_global_inv': s['img_metas'][0]['T_global_inv'],
        })
    print(f"   loaded {len(frames)} frames into GPU memory")

    # Load engines
    print("🔧 Loading engines ...")
    perc_init = TRTInfer(args.engine_perc_init)
    perc_temp = TRTInfer(args.engine_perc_temp)
    mo_init   = TRTInfer(args.engine_mo_init)
    mo_temp   = TRTInfer(args.engine_mo_temp)

    # Pull history sizes from the temporal engine bindings (authoritative)
    nh_det  = perc_temp.inputs['prev_det_feat'].shape[1]
    dim_det = perc_temp.inputs['prev_det_anchor'].shape[2]
    nh_map  = perc_temp.inputs['prev_map_feat'].shape[1]
    dim_map = perc_temp.inputs['prev_map_anchor'].shape[2]
    Q       = mo_temp.inputs['mo_history_anchor'].shape[2]
    print(f"   nh_det={nh_det}, dim_det={dim_det}, nh_map={nh_map}, dim_map={dim_map}, Q={Q}")

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

    def run_frame(frame, history_det, history_map, history_motion, prev_global_mat, prev_time):
        """Run one frame through the cascade. Returns planning trajectory + updated state."""
        ts = frame['timestamp']
        if prev_time is None:
            dt, is_start = 0.5, True
        else:
            dt = ts - prev_time
            is_start = (dt > 2.0 or dt < 0)
        if is_start:
            dt = 0.5
            history_det = zero_hist_det()
            history_map = zero_hist_map()
            history_motion = zero_hist_motion()
            prev_global_mat = None

        dt_tensor = torch.tensor([dt], device='cuda', dtype=torch.float32)
        mask_tensor = torch.tensor([not is_start], device='cuda', dtype=torch.bool)

        if prev_global_mat is None:
            t_mat_np = np.eye(4, dtype=np.float32)
        else:
            t_mat_np = frame['T_global_inv'] @ prev_global_mat
        instance_t_matrix = torch.from_numpy(t_mat_np).float().cuda().unsqueeze(0)

        feed_perc = {
            'img': frame['img'], 'projection_mat': frame['projection_mat'],
            'instance_t_matrix': instance_t_matrix, 'time_interval': dt_tensor,
            **history_det, **history_map,
        }
        engine_perc = perc_init if is_start else perc_temp
        out_p = engine_perc.infer(feed_perc)

        history_det = {
            'prev_det_feat':   out_p['next_det_feat'],
            'prev_det_anchor': out_p['next_det_anchor'],
            'prev_det_conf':   out_p['next_det_conf'],
            'prev_det_id':     out_p['next_det_instance_id'],
            'prev_id_count':   out_p['next_id_count'],
        }
        history_map = {
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
            'instance_t_matrix':          instance_t_matrix,
            'mask':                       mask_tensor,
            **history_motion,
        }
        engine_mo = mo_init if is_start else mo_temp
        out_m = engine_mo.infer(feed_mo)

        history_motion = {
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

        # What the cart consumes: planned trajectory (and we sync to make sure GPU is done)
        planned = out_m['plan_reg']
        torch.cuda.synchronize()
        return planned, history_det, history_map, history_motion, frame['T_global'], ts

    # Warmup
    print(f"🔥 Warming up ({args.warmup} iters) ...")
    hist_d, hist_m, hist_mo = zero_hist_det(), zero_hist_map(), zero_hist_motion()
    pgm, pt = None, None
    for _ in range(args.warmup):
        _, hist_d, hist_m, hist_mo, pgm, pt = run_frame(frames[0], hist_d, hist_m, hist_mo, pgm, pt)

    # Reset before timed run
    hist_d, hist_m, hist_mo = zero_hist_det(), zero_hist_map(), zero_hist_motion()
    pgm, pt = None, None

    # Live loop
    rate = args.rate_hz
    if rate > 0:
        period = 1.0 / rate
        print(f"🎥 Simulating camera at {rate:.1f} Hz over {len(frames)} frames "
              f"({len(frames)/rate:.1f} s of 'video') ...")
    else:
        period = 0.0
        print(f"🏎️  Unbounded mode (max-FPS) over {len(frames)} frames ...")

    latencies_capture = []  # wall-clock capture → output, ms
    latencies_infer = []    # arrival → output, ms (pure inference)
    overruns = 0

    t_sim_start = time.perf_counter()
    for i, frame in enumerate(frames):
        if rate > 0:
            t_capture = t_sim_start + i * period
            now = time.perf_counter()
            if now < t_capture:
                time.sleep(t_capture - now)
            else:
                # Frame arrived before we were ready; we're falling behind.
                pass
        else:
            t_capture = time.perf_counter()

        t_arrive = time.perf_counter()
        _, hist_d, hist_m, hist_mo, pgm, pt = run_frame(frame, hist_d, hist_m, hist_mo, pgm, pt)
        t_done = time.perf_counter()

        latencies_capture.append((t_done - t_capture) * 1000)
        infer_ms = (t_done - t_arrive) * 1000
        latencies_infer.append(infer_ms)
        if rate > 0 and infer_ms > period * 1000:
            overruns += 1

    t_sim_end = time.perf_counter()

    lat_c = np.array(latencies_capture)
    lat_i = np.array(latencies_infer)
    sim_dur = t_sim_end - t_sim_start

    print(f"\n📊 Live-cart simulation report")
    print(f"   frames:                  {len(frames)}")
    if rate > 0:
        print(f"   simulated camera rate:   {rate:.1f} Hz ({period*1000:.0f} ms period)")
        print(f"   simulated duration:      {len(frames)/rate:.2f} s")
    print(f"   wall-clock duration:     {sim_dur:.2f} s")
    print(f"   sustained throughput:    {len(frames)/sim_dur:.1f} FPS")
    if rate > 0:
        print(f"   inference overruns:      {overruns}/{len(frames)} "
              f"(frames slower than the {period*1000:.0f} ms camera period)")
    print()
    print(f"   capture → output latency (ms)   [what downstream sees]:")
    print(f"     mean / median / p95 / p99 / max:")
    print(f"     {lat_c.mean():6.1f} / {np.median(lat_c):6.1f} / "
          f"{np.percentile(lat_c, 95):6.1f} / {np.percentile(lat_c, 99):6.1f} / "
          f"{lat_c.max():6.1f}")
    print()
    print(f"   pure inference time (ms)         [model + sync]:")
    print(f"     mean / median / p95 / p99 / max:")
    print(f"     {lat_i.mean():6.1f} / {np.median(lat_i):6.1f} / "
          f"{np.percentile(lat_i, 95):6.1f} / {np.percentile(lat_i, 99):6.1f} / "
          f"{lat_i.max():6.1f}")
    print(f"   max achievable FPS:      {1000.0 / lat_i.mean():.1f}")


if __name__ == "__main__":
    main()
