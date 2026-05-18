# BEV Predictions JSON Format

This document describes `bev_predictions.json` — a portable export of SparseDrive's
BEV predictions for a 20-second nuScenes clip — and how to render it.

A pre-generated sample is included in `docs/sample_output/`:

- `bev_predictions.json` — predictions for the 40-frame (20s @ 2Hz) clip
- `bev_20s_clip.mp4`     — original visualization (camera + BEV pred + BEV GT)
- `bev_from_json.mp4`    — BEV-only rendering produced from `bev_predictions.json`
                           by `tools/render_bev_from_json.py` (proves the JSON
                           round-trips back to a video without the model)

Generated with:

```bash
python tools/export_bev_json.py \
    work_dirs/sparsedrive_small_stage2/results_trt.pkl \
    work_dirs/sparsedrive_small_stage2/bev_predictions.json \
    --det_score_thr 0.25 --map_score_thr 0.4
```

Rendered with:

```bash
python tools/render_bev_from_json.py \
    work_dirs/sparsedrive_small_stage2/bev_predictions.json \
    work_dirs/sparsedrive_small_stage2/bev_from_json.mp4
```

## File structure

```jsonc
{
  "meta": {
    "format_version": 1,
    "frame_count": 40,
    "frame_rate_hz": 2,
    "frame_convention": "ego_lidar",
    "axes": "x forward (+), y left (+), z up (+); meters",
    "det_score_threshold": 0.25,
    "map_score_threshold": 0.4,
    "traj_horizon_steps": 12,
    "traj_step_seconds": 0.5,
    "plan_horizon_steps": 6,
    "plan_step_seconds": 0.5,
    "det_class_names": ["car", "truck", ..., "traffic_cone"],
    "map_class_names": ["divider", "ped_crossing", "boundary"]
  },
  "frames": [ /* one entry per timestep */ ]
}
```

## Frame entry

```jsonc
{
  "frame_idx": 0,
  "detections":   [ /* dynamic objects */ ],
  "map_elements": [ /* static map polylines */ ],
  "planning":     { "final_xy": [...], "modes_topk": [...] }
}
```

### `detections[]`
Each detection (3D bounding box + motion forecast):

| field | type | meaning |
|---|---|---|
| `id` | int | persistent instance id (stable across frames) |
| `class` / `class_id` | string / int | one of `det_class_names` |
| `score` | float | detection confidence in `[0, 1]` |
| `x`, `y`, `z` | float | box center in ego frame, meters |
| `w`, `l`, `h` | float | width, length, height in meters |
| `yaw` | float | heading angle in radians (CCW from +x) |
| `vx`, `vy` | float | velocity in ego frame, m/s |
| `motion_modes[]` | list | top-K predicted future trajectories |

Each motion mode:
```jsonc
{ "score": 0.51, "xy": [[x0, y0], ..., [x11, y11]] }
```
12 steps × 0.5s = 6 seconds into the future.

### `map_elements[]`
Each static map element:

| field | type | meaning |
|---|---|---|
| `class` / `class_id` | string / int | `divider`, `ped_crossing`, or `boundary` |
| `score` | float | confidence in `[0, 1]` |
| `polyline` | `[[x, y], ...]` | 20 sample points in ego frame, meters |

### `planning`
- `final_xy`: `[[x, y], ...]` — 6 steps × 0.5s = 3 seconds of planned ego motion.
- `modes_topk[]`: top mode candidates, each with `{command, mode, score, xy}`.
  `command` ∈ {0 = turn-left, 1 = straight, 2 = turn-right}; `mode` ∈ [0, 5].

## Coordinate frame

All coordinates are in the **ego/LiDAR frame at the current frame's timestamp**:

- `+x` is forward (direction of vehicle travel)
- `+y` is left
- `+z` is up
- units are **meters**, angles are **radians**

To convert a world-frame `(x, y)` point to a top-down image (BEV) of size `H × W`
with the ego at the center and `RES` meters per pixel:

```python
px = W // 2 - y / RES   # +y is left, so it moves toward the left edge
py = H // 2 - x / RES   # +x is forward, so it moves toward the top
```

## Minimal renderer

`tools/render_bev_from_json.py` is a self-contained ~100-line OpenCV renderer
that draws:

- **gray polygon** — the ego vehicle at the canvas center
- **white / cyan / magenta polylines** — map dividers, ped crossings, boundaries
- **colored rotated rectangles** — detected objects (one color per class)
- **yellow polylines** — top motion forecast for each detection
- **green polyline** — planned ego trajectory

It only depends on `numpy` and `opencv-python` — no PyTorch, TensorRT, or mmcv.

## Reusing the JSON

The JSON is intended to be a **stable handoff format**: downstream consumers
(planning stacks, dashboards, replay tools) can read it without pulling in the
SparseDrive model code or its dependencies.
