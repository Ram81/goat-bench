#!/usr/bin/env python3
"""Visualise and summarise tours produced by collect_tour.py.

Self-contained on purpose: no habitat, no simulator, no GPU. It reads only the
`rgb/`, `depth/` and `poses.npz` a tour directory contains, so coverage can be
inspected on a laptop while collection runs elsewhere.

    python scripts/exploration/view_tour.py data/tours            # every tour
    python scripts/exploration/view_tour.py data/tours/<scene>_<id>  # just one

Writes `tour.mp4` and `trajectory.png` into each tour directory and prints a
coverage summary.
"""
import argparse
import glob
import json
import os

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")  # no display on a headless collection box
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEPTH_SCALE = 1000.0  # stored uint16 millimetres -> metres
CELL_SIZE_M = 0.5  # coverage is counted in cells this wide


def load_depth(path):
    """Read one stored depth frame as metres. 0 means invalid."""
    # Pillow decodes 16-bit PNGs as int32; the stored values are uint16 mm.
    return np.asarray(imageio.imread(path), dtype=np.float32) / DEPTH_SCALE


def colourise_depth(depth, dmax):
    """Blue (near) -> red (far) false colour; invalid pixels stay black."""
    valid = depth > 0
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.any():
        n = np.clip(depth / max(dmax, 1e-6), 0, 1)
        out[..., 0] = (255 * n).astype(np.uint8)
        out[..., 1] = (255 * (1 - np.abs(2 * n - 1))).astype(np.uint8)
        out[..., 2] = (255 * (1 - n)).astype(np.uint8)
        out[~valid] = 0
    return out


def make_video(tour, out_path, fps, stride, with_depth):
    """Write rgb (and optionally depth) side by side. Returns the frame count."""
    rgbs = sorted(glob.glob(f"{tour}/rgb/*.png"))[::stride]
    deps = sorted(glob.glob(f"{tour}/depth/*.png"))[::stride]
    if not rgbs and not deps:
        return 0

    dmax = 1.0
    if deps:
        # Sample ~20 frames rather than all of them; the 90th percentile only
        # needs to be roughly right, and decoding every frame twice is slow.
        sample = deps[:: max(len(deps) // 20, 1)]
        dmax = float(np.percentile([load_depth(p).max() for p in sample], 90))

    # A depth-only tour (collected with --no_rgb) still renders.
    frames = rgbs if rgbs else deps
    with imageio.get_writer(out_path, fps=fps, macro_block_size=None) as writer:
        for i, path in enumerate(frames):
            if rgbs:
                frame = np.asarray(imageio.imread(path))
                if with_depth and i < len(deps):
                    frame = np.concatenate(
                        [frame, colourise_depth(load_depth(deps[i]), dmax)], axis=1
                    )
            else:
                frame = colourise_depth(load_depth(path), dmax)
            writer.append_data(frame)
    return len(frames)


def coverage_stats(xy, theta):
    """Path/coverage summary. Returns (text, dict)."""
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1) if len(xy) > 1 else np.zeros(0)
    dtheta = np.abs(np.rad2deg(np.diff(theta))) if len(theta) > 1 else np.zeros(0)
    forward = step > 0.01
    rotating = (dtheta > 1) & ~forward
    cells = {
        (int(np.floor(a / CELL_SIZE_M)), int(np.floor(b / CELL_SIZE_M)))
        for a, b in xy
    }

    stats = {
        "frames": int(len(xy)),
        "path_length_m": float(step.sum()),
        "extent_x_m": float(np.ptp(xy[:, 0])),
        "extent_y_m": float(np.ptp(xy[:, 1])),
        "forward_steps": int(forward.sum()),
        "rotation_steps": int(rotating.sum()),
        "distinct_cells": len(cells),
        "area_m2": len(cells) * CELL_SIZE_M**2,
        "views_per_position": float(len(xy) / max(len(cells), 1)),
    }
    pct = lambda a: 100 * a.mean() if len(a) else 0.0  # noqa: E731
    text = (
        f"frames               {stats['frames']}\n"
        f"path length          {stats['path_length_m']:.1f} m\n"
        f"bbox extent          {stats['extent_x_m']:.1f} x {stats['extent_y_m']:.1f} m\n"
        f"forward steps        {stats['forward_steps']} ({pct(forward):.0f}%)\n"
        f"rotation steps       {stats['rotation_steps']} ({pct(rotating):.0f}%)\n"
        f"distinct {CELL_SIZE_M} m cells  {stats['distinct_cells']}"
        f"  (~{stats['area_m2']:.1f} m2)\n"
        f"views per position   {stats['views_per_position']:.1f}"
    )
    return text, stats


def plot_trajectory(tour, out_path):
    """Trajectory plot plus a coverage summary panel. Returns (text, dict)."""
    poses = np.load(f"{tour}/poses.npz")
    xy, theta = poses["gps"][:, :2], poses["compass"][:, 0]
    text, stats = coverage_stats(xy, theta)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].plot(xy[:, 0], xy[:, 1], "-", lw=1, alpha=0.6, color="tab:blue")
    ax[0].scatter(xy[:, 0], xy[:, 1], c=np.arange(len(xy)), cmap="viridis", s=9)
    ax[0].scatter(
        *xy[0], c="lime", s=140, marker="*", zorder=5, label="start", edgecolor="k"
    )
    ax[0].scatter(
        *xy[-1], c="red", s=90, marker="X", zorder=5, label="end", edgecolor="k"
    )
    # Subsample the heading arrows; one per frame is unreadable.
    s = max(len(xy) // 60, 1)
    ax[0].quiver(
        xy[::s, 0], xy[::s, 1], np.cos(theta)[::s], np.sin(theta)[::s],
        color="grey", alpha=0.6, width=0.003, scale=30,
    )
    ax[0].set_aspect("equal")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[0].set_title("trajectory (colour = time, arrows = heading)")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")

    ax[1].axis("off")
    ax[1].text(0.02, 0.98, text, va="top", family="monospace", fontsize=12)
    ax[1].set_title("coverage summary")

    title = os.path.basename(tour.rstrip("/"))
    meta_path = os.path.join(tour, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            reason = json.load(f).get("termination_reason")
        if reason:
            title = f"{title}   (ended: {reason})"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return text, stats


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("tour", help="a tour directory, or a parent of tour directories")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--no_depth", action="store_true", help="omit the depth panel")
    ap.add_argument("--no_video", action="store_true", help="stats and plot only")
    a = ap.parse_args(argv)

    is_tour = os.path.exists(os.path.join(a.tour, "poses.npz"))
    tours = (
        [a.tour]
        if is_tour
        else sorted(
            d for d in glob.glob(f"{a.tour}/*")
            if os.path.exists(os.path.join(d, "poses.npz"))
        )
    )
    if not tours:
        raise SystemExit(f"no tours found under {a.tour}")

    summary = {}
    for tour in tours:
        name = os.path.basename(tour.rstrip("/"))
        print(f"=== {name}")
        text, stats = plot_trajectory(tour, f"{tour}/trajectory.png")
        print(text)
        if not a.no_video:
            n = make_video(tour, f"{tour}/tour.mp4", a.fps, a.stride, not a.no_depth)
            print(f"  wrote tour.mp4 ({n} frames) + trajectory.png")
        summary[name] = stats
        print()

    if len(tours) > 1:
        areas = [s["area_m2"] for s in summary.values()]
        lengths = [s["path_length_m"] for s in summary.values()]
        print(f"--- {len(tours)} tours")
        print(f"area covered   mean {np.mean(areas):.1f} m2   total {np.sum(areas):.1f} m2")
        print(f"path length    mean {np.mean(lengths):.1f} m   total {np.sum(lengths):.1f} m")
        out = os.path.join(a.tour, "coverage.json")
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
