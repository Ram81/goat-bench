# Frontier-exploration coverage tours

Drives an agent around a scene to *cover* it — no goal, no object search — and
records what it saw as `rgb/`, `depth/` and poses. Ported from home-robot's OVMM
exploration stack, with the object-goal, perception and manipulation machinery
removed.

Use it to generate tours over the GOAT-Bench scenes for pretraining, mapping
experiments, or anything else that wants dense observation/pose data rather than
task episodes.

## Status

Verified end-to-end on **MP3D** scenes (Habitat scene loading, the map update,
frontier selection, planning, recording, and visualisation all run and produce
sensible coverage). **Not yet run on HM3D**, because the HM3D scenes and the
GOAT-Bench episode files are not downloadable without accepting the Matterport
terms — see [Data](#data). Switching to HM3D is a config change, not a code
change, but treat the first HM3D run as unverified.

Behaviour was cross-checked against the home-robot tours this was ported from.
Over comparable episodes the two produce near-identical statistics:

| | home-robot (HSSD, 501 frames) | this port (MP3D, 401 frames) |
| --- | --- | --- |
| path length | 49.3 m | 45.4 m |
| area covered | 19.6 m² | 17.8 m² |
| forward / rotation split | 41–44% / 51–53% | 42–51% / 44–51% |
| views per position | 5.4–6.0 | 5.4–6.0 |

## Setup

```bash
./setup_explore_env.sh          # python 3.9 + habitat-sim 0.2.5 + habitat-lab
```

This is **not** goat-bench's `setup.sh`. Exploration does not need the
goal-conditioned model stack (CLIP, LAVIS, VC-1), and it runs on habitat-lab
0.2.5 rather than the 0.2.3 that GOAT training targets. Consequence: in this
environment `goat_bench.exploration` works and GOAT training does not. The
package guards those imports and warns at import time; see
`goat_bench/__init__.py`. To do both, keep two environments.

If you already have a working habitat environment, you can skip the script
entirely — the exploration package only needs habitat, torch, scikit-fmm,
scikit-image, scikit-learn, opencv, imageio, matplotlib, trimesh and
numpy-quaternion.

## Data

Needs the HM3D scenes plus the GOAT-Bench episode files, laid out as:

```
data/
  scene_datasets/hm3d/...                       # HM3D v0.2 scenes
  datasets/goat_bench/v1/{split}/{split}.json.gz  # GOAT-Bench episodes
```

The episode files supply the scene list and the start poses, which is what makes
these *the GOAT-Bench scenes and splits* rather than an arbitrary walk through
HM3D. HM3D requires accepting Matterport's terms; see the main README.

Budget roughly 10 GB for the HM3D val split and ~80 GB for train.

To explore scenes without the GOAT episode files, override the dataset:

```bash
./run_explore_tour.sh \
  habitat.dataset.type=PointNav-v1 \
  "habitat.dataset.data_path='data/datasets/pointnav/hm3d/v1/{split}/{split}.json.gz'"
```

## Usage

```bash
./run_explore_tour.sh                             # 5 episodes, 500 steps each
EPISODES=50 STEPS=750 OUT=data/tours ./run_explore_tour.sh
./run_explore_tour.sh habitat.dataset.split=train   # habitat overrides pass through
```

Then visualise — this needs no simulator and no GPU, so it runs anywhere the
tour directory does:

```bash
python scripts/exploration/view_tour.py data/tours
```

It writes `tour.mp4` (rgb beside false-coloured depth) and `trajectory.png`
(path, headings, coverage summary) into each tour directory, plus a
`coverage.json` across all of them.

Quoting note: habitat `data_path` values contain `{split}`, and hydra's override
grammar rejects braces unless the value is quoted *inside* the argument —
`"habitat.dataset.data_path='.../{split}/{split}.json.gz'"`.

## Output layout

```
data/tours/<scene>_<episode_id>/
  rgb/000000.png ...        # uint8 RGB           (omit with --no_rgb)
  depth/000000.png ...      # uint16 millimetres, 0 = invalid
  poses.npz                 # gps (N,2) metres, compass (N,1) radians
  meta.json                 # scene_id, episode_id, num_frames, termination_reason
data/tours/manifest.json    # all episodes in the run
```

Depth is stored as uint16 millimetres (max 65.535 m) rather than float32: it is
the bulk of the data, and this halves it losslessly at the sensor's precision.
Frames are written as they arrive rather than buffered — an episode's worth of
raw observations is several GB in RAM.

## How it works

Each step:

1. **Map.** Depth is projected into a 2D occupancy grid using the agent's pose,
   giving `obstacle` and `explored` channels.
2. **Frontier.** The explored region is dilated and its border taken; those
   cells — the boundary between known and unknown — become the goal.
3. **Plan.** A fast-marching (FMM) geodesic distance field over the traversable
   map yields a short-term goal, which becomes one discrete action.

Episodes start with a 360° spin so the first frontier is chosen from a full
panorama rather than a single camera frustum.

An episode ends when the frontier is exhausted (`fully_explored` — the scene is
covered), the step budget runs out (`max_steps`), or the agent stops moving
(`stuck`). The reason is recorded in `meta.json`.

The `stuck` check exists because a goal-driven planner's "nowhere left to go"
STOP sits behind a `found_goal` test that never fires during exploration; without
it, a wedged agent burns its whole budget emitting identical frames. That was a
real failure mode in the home-robot original.

## Layout

| Path | What |
| --- | --- |
| `goat_bench/exploration/agent.py` | the agent: map → frontier → plan → action, and termination |
| `goat_bench/exploration/frontier_policy.py` | frontier selection |
| `goat_bench/exploration/module.py` | map update + policy as one `nn.Module` |
| `goat_bench/exploration/habitat_env.py` | habitat adapter: GPS frame, depth encoding, navmesh |
| `goat_bench/exploration/config.py` | all tunables, with the home-robot defaults |
| `goat_bench/exploration/_vendor/` | numerics vendored from home-robot — see its README |
| `config/tasks/explore_hm3d.yaml` | the habitat task: rgbd + gps/compass, no goals |
| `scripts/exploration/collect_tour.py` | collection entry point |
| `scripts/exploration/view_tour.py` | offline visualisation and coverage stats |

## Notes and gotchas

- **Mapping uses a 3.5 m depth cutoff**, not the sensor's 10 m
  (`semantic_map.map_max_depth`). Far returns are noisy and smear obstacles
  across cells the agent cannot see into. home-robot inherits this as an unstated
  default; it is explicit here because it materially changes the map.
- **The planner's footprint radius is 0.05 m**, much smaller than the Stretch's
  0.17 m body (`ExplorationConfig.radius`). This is planning inflation, not
  collision: at the true body radius the planner refuses narrow doorways. The
  simulator still enforces the real radius.
- **The navmesh is recomputed per scene** for GOAT's agent body (height 1.41,
  radius 0.17, max climb 0.20). A scene's shipped navmesh is built for whatever
  agent its authors used, and traversability — which doorways count as passable —
  depends on that. `GOATSim-v0` does this too, but through a habitat-sim 0.2.3
  call signature, so `habitat_env.py` reimplements it version-tolerantly.
- **`allow_sliding` is off.** Sliding along walls lets the agent make progress
  through a collision, which hides obstacles from the map's collision correction.
- Coverage in `view_tour.py` counts distinct 0.5 m cells *visited*, which is a
  proxy for observed area, not the mapped free space.
