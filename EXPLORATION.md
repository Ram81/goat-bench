# Frontier-exploration coverage tours

Drives an agent around a scene to *cover* it — no goal, no object search — and
records what it saw as `rgb/`, `depth/` and poses. Ported from home-robot's OVMM
exploration stack, with the object-goal, perception and manipulation machinery
removed.

Use it to generate tours over the GOAT-Bench scenes for pretraining, mapping
experiments, or anything else that wants dense observation/pose data rather than
task episodes.

## Status

Verified end-to-end on **MP3D** scenes, in goat-bench's own environment and
inside `GOATSim-v0`: scene loading, the per-scene navmesh rebuild, the map
update, frontier selection, planning, recording and visualisation all run and
produce sensible coverage, across single- and multi-scene runs.

**Not yet run on HM3D**, because the HM3D scenes and the GOAT-Bench episode
files are not downloadable without accepting the Matterport terms — see
[Data](#data). Switching to HM3D is a config change, not a code change, but
treat the first HM3D run as unverified.

Behaviour was cross-checked against the home-robot tours this was ported from.
Over comparable episodes the two produce near-identical statistics:

| | home-robot (HSSD, 501 frames) | this port (MP3D, 4 × 401 frames) |
| --- | --- | --- |
| path length | 49.3 m | 42.0 m (38.3–45.9) |
| area covered | 19.6 m² | 17.7 m² (15.8–19.3) |
| forward / rotation split | 41–44% / 51–53% | 40–48% / 45–54% |
| views per position | 5.4–6.0 | 5.2–6.4 |

Views-per-position is the tell-tale: the agent takes roughly six observations
per half-metre cell it visits, in both, which is what the 360° panorama plus
frontier re-planning produces. A tour that comes out far above that range did
not explore — see the last of the [notes](#notes-and-gotchas).

## Setup

```bash
./setup_explore_env.sh          # conda env `goat`: habitat-sim 0.2.3 + habitat-lab v0.2.3
```

This builds **goat-bench's own environment** — the stack the main README
describes — so exploration and GOAT training live in one place. It has to be
that stack: exploration runs inside `GOATSim-v0`, whose navmesh recomputation
calls `recompute_navmesh(..., include_static_objects=False)`, a keyword
habitat-sim removed in 0.2.4.

Two deviations from the README's recipe, both forced rather than chosen:

- **python 3.8, not 3.7.** Nothing in goat-bench needs 3.7; torch needs 3.8.
  Ada GPUs (RTX 40-series) want a CUDA 11.8 build, torch first shipped one in
  2.0, and torch 2.x dropped python 3.7. habitat-sim 0.2.3 publishes py3.8
  builds, so this costs nothing. On pre-Ada hardware the README's
  `python=3.7` + `cudatoolkit=11.3` works as written.
- **A few version pins** that the 2024-era requirements no longer resolve to on
  their own. Without them the install succeeds and then fails at import:
  `opencv-python==4.8.1.78` (≥4.9 changed `applyColorMap`'s output shape, which
  breaks `import habitat` itself), `sophuspy==0.0.8` (last release exposing the
  module as `sophus`), `lmdb==1.4.1` (newer wheels are built against a newer
  CPython ABI), and `faster_fifo` (without it habitat-baselines takes a
  fallback path that is itself broken). Each is commented in the script.

The script leaves out the goal-conditioned model stack (CLIP, LAVIS, VC-1) —
several GB that a coverage tour never loads. `goat_bench/__init__.py` guards
those imports and warns at import time, so exploration runs without them. To
train GOAT from the same environment, add them:

```bash
pip install -r requirements.txt git+https://github.com/openai/CLIP.git
```

If you already have goat-bench's environment, you do not need the script — just
add what exploration itself uses: `scikit-fmm` (the fast-marching planner),
`scikit-image`, `sophuspy==0.0.8`, `imageio`, `imageio-ffmpeg`, `matplotlib`
and `trimesh`.

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
  depends on that, so this is what makes the agent move through the same free
  space GOAT does. `GOATSim-v0` does it, on construction and on every scene
  change. `habitat_env.py` carries an equivalent implementation for running on
  habitat-sim 0.2.5, where `GOATSim-v0` cannot be constructed; `collect_tour.py`
  uses it only when the simulator is not `GOATSim-v0`, so the work is never
  done twice.
- **`allow_sliding` is off.** Sliding along walls lets the agent make progress
  through a collision, which hides obstacles from the map's collision correction.
- Coverage in `view_tour.py` counts distinct 0.5 m cells *visited*, which is a
  proxy for observed area, not the mapped free space.
- **A start the agent cannot leave burns the whole budget.** If an episode
  begins in a closet, on a balcony, or on any patch of floor cut off from the
  rest of the scene, frontier cells still exist beyond the walls but none are
  reachable, so the planner keeps re-aiming at them and the agent turns on the
  spot. Neither terminator fires: `fully_explored` needs an *empty* frontier,
  and `stuck` counts a heading change as movement (`agent.py`,
  `note_pose`), which rotating in place is. It shows up in `view_tour.py` as a
  high views-per-position against very few forward steps — 18 views/position
  and 55 forward steps out of 400, in one observed case, versus ~5.5 and ~190
  for a healthy tour. If you generate your own episodes, sample start poses
  from the navmesh *as rebuilt for the Stretch body*, not the one the scene
  ships, and keep starts on a large connected island
  (`pathfinder.island_radius`). The GOAT-Bench episodes already satisfy this.
