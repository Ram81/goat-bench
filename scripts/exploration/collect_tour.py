#!/usr/bin/env python3
"""Collect frontier-exploration coverage tours in GOAT-Bench scenes.

Each episode drives the agent from an episode start pose until the frontier is
exhausted, streaming rgb / depth / pose to disk as it goes. Frames are written
as they arrive rather than accumulated: an Observations object is several MB, so
buffering a 500-step episode and pickling it at the end costs gigabytes of RAM
for data that is mostly redundant.

    python scripts/exploration/collect_tour.py --out data/tours --num_episodes 5

Trailing arguments are habitat config overrides, e.g.

    ... --num_episodes 2 habitat.dataset.split=train
"""
import argparse
import json
import os

import imageio.v2 as imageio
import numpy as np

# Keep BLAS single-threaded: the map update is small and per-step, and thread
# pool contention costs more than it saves.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import habitat  # noqa: E402
from habitat.config.default_structured_configs import (  # noqa: E402
    register_hydra_plugin,
)

from goat_bench.config import HabitatConfigPlugin  # noqa: E402
from goat_bench.exploration import (  # noqa: E402
    DiscreteNavigationAction,
    ExplorationConfig,
    FrontierExplorationAgent,
    HabitatExplorationEnv,
)

DEPTH_SCALE = 1000.0  # metres -> millimetres, stored as uint16 (max 65.535 m)
# Sentinels written by HabitatExplorationEnv for pixels the sensor did not
# return; far beyond any real reading, so they are easy to exclude.
DEPTH_INVALID_THRESHOLD = 100.0
MAX_STORED_DEPTH_M = 65.0


def save_frame(out_dir, index, obs, save_rgb=True):
    """Write one rgb/depth pair and return the pose entry for that frame."""
    if save_rgb:
        imageio.imwrite(
            os.path.join(out_dir, "rgb", f"{index:06d}.png"),
            np.asarray(obs.rgb).astype(np.uint8),
        )
    depth = np.asarray(obs.depth, dtype=np.float32)
    # 0 encodes "invalid" in the stored PNG, which is why genuine zero-range
    # readings and sensor drop-outs are collapsed together here.
    depth = np.where(
        depth >= DEPTH_INVALID_THRESHOLD, 0.0, np.clip(depth, 0.0, MAX_STORED_DEPTH_M)
    )
    imageio.imwrite(
        os.path.join(out_dir, "depth", f"{index:06d}.png"),
        (depth * DEPTH_SCALE).astype(np.uint16),
    )
    return {
        "gps": np.asarray(obs.gps, dtype=np.float32),
        "compass": np.asarray(obs.compass, dtype=np.float32),
    }


def episode_key(episode):
    """A filesystem-safe id: <scene>_<episode_id>."""
    scene = os.path.basename(episode.scene_id).split(".")[0]
    return f"{scene}_{episode.episode_id}"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default="data/tours", help="output directory")
    p.add_argument("--num_episodes", type=int, default=2)
    p.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="step budget per episode; an episode may end earlier once the "
        "frontier is exhausted",
    )
    p.add_argument(
        "--stuck_patience",
        type=int,
        default=30,
        help="end the episode after this many steps with no pose change (0=off)",
    )
    p.add_argument(
        "--config",
        default="config/tasks/explore_hm3d.yaml",
        help="habitat task config. `habitat.get_config` tries the path as given "
        "first (so this is relative to the working directory, i.e. the repo "
        "root) and only then relative to habitat-lab's own config directory",
    )
    p.add_argument(
        "--device_id", type=int, default=0, help="CUDA device for the map update; -1 for CPU"
    )
    p.add_argument(
        "--save_rgb",
        dest="save_rgb",
        action="store_true",
        default=True,
        help="write rgb frames (default)",
    )
    p.add_argument(
        "--no_rgb",
        dest="save_rgb",
        action="store_false",
        help="skip rgb frames; depth and poses are enough to rebuild coverage",
    )
    p.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="habitat config overrides, e.g. habitat.dataset.split=train",
    )
    return p.parse_args(argv)


def build_agent(habitat_config, args):
    """Construct the agent with sensor geometry taken from the live sim config."""
    cfg = ExplorationConfig()
    sim = habitat_config.habitat.simulator
    depth_sensor = sim.agents.main_agent.sim_sensors.depth_sensor

    # The map module's projection is only correct if these match the simulator
    # exactly, so they are read back rather than duplicated in the dataclass.
    cfg.environment.frame_height = depth_sensor.height
    cfg.environment.frame_width = depth_sensor.width
    cfg.environment.hfov = float(depth_sensor.hfov)
    cfg.environment.camera_height = float(depth_sensor.position[1])
    cfg.environment.min_depth = float(depth_sensor.min_depth)
    cfg.environment.max_depth = float(depth_sensor.max_depth)
    cfg.environment.turn_angle = float(sim.turn_angle)
    cfg.environment.forward = float(sim.forward_step_size)

    return FrontierExplorationAgent(
        cfg,
        device_id=args.device_id,
        max_steps=args.max_steps,
        stuck_patience=args.stuck_patience,
    ), cfg


def collect_episode(env, agent, obs, out_dir, args):
    """Run one already-reset episode to termination.

    Takes the first observation rather than resetting itself: `env.reset()`
    advances the episode iterator, so resetting again here would silently skip
    every other episode.

    Returns (num_frames, reason).
    """
    subdirs = ("rgb", "depth") if args.save_rgb else ("depth",)
    for sub in subdirs:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    agent.reset()

    poses = [save_frame(out_dir, 0, obs, args.save_rgb)]
    index, reason, done = 0, "episode_over", False

    while not done:
        action, info = agent.act(obs)
        if action == DiscreteNavigationAction.STOP:
            reason = info.get("termination_reason") or "agent_stop"
            break
        obs, done = env.apply_action(action)
        # The agent needs the post-action pose to tell "turning in place" from
        # "wedged against geometry".
        agent.note_pose(obs)
        index += 1
        poses.append(save_frame(out_dir, index, obs, args.save_rgb))

    np.savez_compressed(
        os.path.join(out_dir, "poses.npz"),
        **{k: np.stack([p[k] for p in poses]) for k in poses[0]},
    )
    return index + 1, reason


def main(argv=None):
    args = parse_args(argv)

    register_hydra_plugin(HabitatConfigPlugin)
    overrides = [o for o in (args.overrides or []) if o != "--"]
    habitat_config = habitat.get_config(args.config, overrides=overrides)

    agent, cfg = build_agent(habitat_config, args)
    print(
        f"[config] {cfg.environment.frame_height}x{cfg.environment.frame_width} "
        f"hfov={cfg.environment.hfov} camera_height={cfg.environment.camera_height} "
        f"turn={cfg.environment.turn_angle}deg map_depth<={cfg.semantic_map.map_max_depth}m",
        flush=True,
    )

    sim_config = habitat_config.habitat.simulator
    depth_sensor = sim_config.agents.main_agent.sim_sensors.depth_sensor

    # Rebuild each scene's navmesh for this agent body, as GOAT-Bench does.
    #
    # GOATSim-v0 already does this itself, on construction and on every scene
    # change, so with the default config this stays None and the simulator is
    # left to it. The wrapper's copy is the fallback for habitat-sim 0.2.5,
    # where GOATSim cannot be constructed at all (see the task config) and the
    # stock Sim-v0 is used instead. Without either, the scene's shipped navmesh
    # is used as-is -- built for whatever agent the dataset authors used.
    navmesh_settings = None
    if sim_config.type != "GOATSim-v0" and "navmesh_settings" in sim_config:
        navmesh_settings = {
            "agent_height": float(sim_config.agents.main_agent.height),
            "agent_radius": float(sim_config.agents.main_agent.radius),
            "agent_max_climb": float(sim_config.navmesh_settings.agent_max_climb),
            "cell_height": float(sim_config.navmesh_settings.cell_height),
        }

    env = HabitatExplorationEnv(
        habitat.Env(config=habitat_config),
        min_depth=float(depth_sensor.min_depth),
        max_depth=float(depth_sensor.max_depth),
        normalize_depth=bool(depth_sensor.normalize_depth),
        navmesh_settings=navmesh_settings,
    )

    os.makedirs(args.out, exist_ok=True)
    manifest = []
    try:
        for ep in range(args.num_episodes):
            try:
                obs = env.reset()  # advances the episode iterator
            except StopIteration:
                print(f"episode iterator exhausted after {ep} episodes", flush=True)
                break
            episode = env.get_current_episode()
            out_dir = os.path.join(args.out, episode_key(episode))

            num_frames, reason = collect_episode(env, agent, obs, out_dir, args)
            meta = {
                "scene_id": episode.scene_id,
                "episode_id": episode.episode_id,
                "num_frames": num_frames,
                "termination_reason": reason,
                "max_steps": args.max_steps,
            }
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            manifest.append(meta)
            print(
                f"[{ep + 1}/{args.num_episodes}] {os.path.basename(out_dir)}: "
                f"{num_frames} frames ({reason}) -> {out_dir}",
                flush=True,
            )
    finally:
        env.close()

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(manifest)} tours to {args.out}", flush=True)


if __name__ == "__main__":
    main()
