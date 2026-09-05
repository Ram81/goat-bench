# Vendored from home-robot

These modules are copied from home-robot (MIT, Copyright (c) Meta Platforms,
Inc. and affiliates), from the `facebookresearch/home-robot` repository at
commit `ede6a67a`.

They are vendored rather than reimplemented because their behaviour is tuned:
the projection thresholds, obstacle dilation radii and short-term-goal logic are
what make frontier exploration actually cover a scene instead of oscillating in
a doorway. Rewriting them invites silent regressions that only show up as worse
coverage many episodes later.

| File | Source in home-robot (`src/home_robot/home_robot/`) |
| --- | --- |
| `depth.py` | `utils/depth.py` |
| `pose.py` | `utils/pose.py` |
| `rotation.py` | `utils/rotation.py` |
| `morphology.py` | `utils/morphology.py` |
| `spot.py` | `utils/spot.py` |
| `geometry.py` | `utils/geometry/_base.py` |
| `map_utils.py` | `mapping/map_utils.py` |
| `semantic_map_state.py` | `mapping/semantic/categorical_2d_semantic_map_state.py` |
| `semantic_map_module.py` | `mapping/semantic/categorical_2d_semantic_map_module.py` |
| `fmm_planner.py` | `navigation_planner/fmm_planner.py` |
| `discrete_planner.py` | `navigation_planner/discrete_planner.py` |
| `constants.py` | merge of `utils/constants.py` + `mapping/semantic/constants.py` |
| `interfaces.py` | trimmed stand-in for `core/interfaces.py` |

## Changes made

Kept as close to verbatim as possible. The edits are:

1. `home_robot.*` imports rewritten to package-relative imports.
2. `InstanceMemory` import dropped and its type hints widened to `Optional[Any]`.
   Instance tracking is reached only when `record_instance_ids` or
   `evaluate_instance_tracking` is set, and coverage exploration sets neither.
3. The Open3D `show_point_cloud` debug import (reachable only under
   `debug_mode`) replaced with a `NotImplementedError`.
4. `constants.py` and `interfaces.py` written fresh to close the dependency on
   the rest of `home_robot`.
5. `semantic_map_module.py`: `seq_camera_poses[:, t]` in `forward` guarded for
   `None`. `_update_local_map_and_pose` already documents and implements a
   `camera_pose is None` path (level camera at the configured height, which is
   what a Habitat agent restricted to move/turn actions has), but the caller
   made it unreachable. One line; no change when a camera pose is supplied.

No numerical changes, and no control-flow changes beyond item 5. When updating against a newer home-robot,
re-copy and reapply these four edits rather than hand-merging.

## What is unused here

`semantic_map_module.py` still carries the semantic and instance channels, and
`spot.py` supports the `hull` / `gaze` exploration types. Coverage exploration
uses none of them (`num_sem_categories=1`, `exploration_type="default"`). They
are retained so the file stays a clean copy of upstream.
