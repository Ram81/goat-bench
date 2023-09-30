import os
import shutil
from typing import Any, Dict, List, Optional

import cv2
import imageio
import numpy as np
import torch
from habitat.tasks.utils import compute_pixel_coverage
from habitat_sim._ext.habitat_sim_bindings import SemanticObject
from habitat_sim.agent.agent import AgentState
from habitat_sim.simulator import Simulator
from habitat_sim.utils.common import d3_40_colors_rgb
from numpy import ndarray
from PIL import Image
from torchvision.ops import masks_to_boxes
from torchvision.transforms import PILToTensor
from torchvision.utils import draw_bounding_boxes

from goat.dataset.pose_sampler import PoseSampler

IMAGE_DIR = "data/images/ovon_dataset_gen/debug"
MAX_DIST = [0, 0, 200]  # Blue
NON_NAVIGABLE = [150, 150, 150]  # Grey
POINT_COLOR = [150, 150, 150]  # Grey
VIEW_POINT_COLOR = [0, 200, 0]  # Green
CENTER_POINT_COLOR = [200, 0, 0]  # Red
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def short_scene(scene):
    return scene.split("/")[-1].split(".")[0]


color2RGB = {
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Red": (255, 0, 0),
    "Lime": (0, 255, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Cyan": (0, 255, 255),
    "Magenta": (255, 0, 255),
    "Silver": (192, 192, 192),
    "Gray": (128, 128, 128),
    "Maroon": (128, 0, 0),
    "Olive": (128, 128, 0),
    "Green": (0, 128, 0),
    "Purple": (128, 0, 128),
    "Teal": (0, 128, 128),
    "Navy": (0, 0, 128),
}


def get_depth(depth_obs, semantic_obs, objects):
    obj_depths = []
    for obj in objects:
        id = obj.semantic_id
        depth = np.mean(depth_obs[semantic_obs == id])
        obj_depths.append("{:.2f}".format(depth))

    return obj_depths


def get_color(rgb_obs, semantic_obs, objects):
    """
    Returns color name or None if object does not have specific color
    """
    colors = np.array(list(color2RGB.values()))
    obj_colors = []
    for obj in objects:
        id = obj.semantic_id
        rgb = rgb_obs[semantic_obs == id][:, :3]
        color_ids = np.argmin(
            np.linalg.norm(rgb[:, np.newaxis, :] - colors, axis=2),
            axis=1,
        )
        maj_color = np.bincount(color_ids).argmax()
        if (color_ids[color_ids == maj_color]).shape[0] / color_ids.shape[
            0
        ] > 0.5:
            obj_colors.append(list(color2RGB.keys())[maj_color])
        else:
            obj_colors.append(None)
    return obj_colors


def obs_to_frame(obs):
    rgb = cv2.cvtColor(obs["color_sensor"], cv2.COLOR_BGRA2RGB)
    dmap = (obs["depth_sensor"] / 10 * 255).astype(np.uint8)
    dmap_colored = cv2.applyColorMap(dmap, cv2.COLORMAP_VIRIDIS)

    semantic_obs = obs["semantic_sensor"]
    semantic_img = Image.new(
        "P", (semantic_obs.shape[1], semantic_obs.shape[0])
    )
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    return np.concatenate([rgb, dmap_colored, semantic_img], axis=1)


def save_candidate_imgs(
    obs: List[Dict[str, ndarray]],
    frame_covs: List[float],
    save_to: str,
) -> None:
    """Write coverage stats on candidate images and save all to disk"""
    os.makedirs(save_to, exist_ok=True)

    font = cv2.FONT_HERSHEY_SIMPLEX
    loc = (50, 50)
    lt = cv2.LINE_AA
    w = (255, 255, 255)
    b = (0, 0, 0)

    for i, (o, fc) in enumerate(zip(obs, frame_covs)):
        txt = f"Frame coverage: {round(fc, 2)}"
        img = obs_to_frame(o)
        img = cv2.putText(img, txt, loc, font, 1, b, 4, lt)
        img = cv2.putText(img, txt, loc, font, 1, w, 2, lt)
        cv2.imwrite(os.path.join(save_to, f"candidate_{i}.png"), img)


def get_bounding_box(
    obs: List[Dict[str, ndarray]],
    objectList: List[SemanticObject],
    target: Dict[str, Any],
    depths: ndarray = None,
    bbox_color: str = "red",
):
    """Return the image with bounding boxes drawn on objects inside objectList"""
    N, H, W = (
        len(objectList),
        obs["semantic_sensor"].shape[0],
        obs["semantic_sensor"].shape[1],
    )
    masks = np.zeros((N, H, W))
    for i, obj in enumerate(objectList):
        masks[i] = (
            obs["semantic_sensor"] == np.array([[(obj.semantic_id)]])
        ).reshape((1, H, W))

    boxes = masks_to_boxes(torch.from_numpy(masks))

    bbox_metadata = []
    added_wall = False
    for i, obj in enumerate(objectList):
        if (
            obj.category.name() == target.category.name()
            and obj.semantic_id != target.semantic_id
            and obj.id != target.id
        ):
            continue
        if "wall" in obj.category.name() and added_wall:
            continue
        if "wall" in obj.category.name():
            added_wall = True
        bbox_metadata.append(
            {
                "category": obj.category.name(),
                "semantic_id": obj.semantic_id,
                "bbox": boxes[i].cpu().detach().numpy().tolist(),
                "area": float(
                    ((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
                    .cpu()
                    .detach()
                    .numpy()
                    / (H * W)
                ),
                "is_target": obj.semantic_id == target.semantic_id,
            }
        )

    area = []
    for box in boxes:
        area.append(
            ((box[2] - box[0]) * (box[3] - box[1])).cpu().detach().numpy()
            / (H * W)
        )
    rgb_key = "color" if "color" in obs.keys() else "rgb"
    img = Image.fromarray(obs["color_sensor"][:, :, :3], "RGB")
    if depths is None:
        labels = [
            f"{obj.category.name()}_{obj.semantic_id}"
            for i, obj in enumerate(objectList)
        ]
    else:
        labels = [
            f"{obj.category.name()}_{obj.semantic_id}_d = {depths[i]}"
            for i, obj in enumerate(objectList)
        ]
    drawn_img = draw_bounding_boxes(
        PILToTensor()(img),
        boxes,
        colors=bbox_color,
        width=2,
        labels=labels,
        font_size=10,
    )
    boxes = boxes.cpu().detach().numpy()
    return drawn_img, bbox_metadata, area


def draw_bbox_on_img(observation, boxes, labels, bbox_color="red"):
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.Tensor(boxes)

    img = Image.fromarray(observation["color_sensor"][:, :, :3], "RGB")

    drawn_img = draw_bounding_boxes(
        PILToTensor()(img),
        boxes,
        colors=bbox_color,
        width=2,
        labels=labels,
        font_size=10,
    )
    return drawn_img


def _get_iou_pose(
    sim: Simulator, pose: AgentState, objectList: List[SemanticObject]
):
    """Get coverage of all the objects in the objectList"""
    agent = sim.get_agent(0)
    agent.set_state(pose)
    obs = sim.get_sensor_observations()
    cov = np.zeros((len(objectList), 1))
    for i, obj in enumerate(objectList):
        cov_obj = compute_pixel_coverage(obs["semantic"], obj.semantic_id)
        if cov_obj <= 0:
            return None, None, "Failure: All Objects are not Visualized"
        cov[i] = cov_obj
    return cov, pose, "Successs"


def get_best_viewpoint_with_posesampler(
    sim: Simulator,
    pose_sampler: PoseSampler,
    objectList: List[SemanticObject],
):
    search_center = np.mean(
        np.array([obj.aabb.center for obj in objectList]), axis=0
    )
    candidate_states = pose_sampler.sample_agent_poses_radially(search_center)
    candidate_poses_ious = list(
        _get_iou_pose(sim, pos, objectList) for pos in candidate_states
    )
    candidate_poses_ious_filtered = [
        p for p in candidate_poses_ious if (p[0] is not None)
    ]
    candidate_poses_sorted = sorted(
        candidate_poses_ious_filtered, key=lambda x: np.sum(x[0]), reverse=True
    )
    if candidate_poses_sorted:
        return True, candidate_poses_sorted[0]
    else:
        return False, None


def objects_in_view(observation, target_obj, threshold=0.005, max_depth=4.5):
    depth_obs = observation["depth_sensor"]
    semantic_obs = observation["semantic_sensor"]

    area = np.prod(semantic_obs.shape)
    obj_ids, num_pixels_per_obj = np.unique(semantic_obs, return_counts=True)
    objects = []
    depth_filtered = []

    avg_depths = []
    for obj_id, total_pixels in zip(obj_ids, num_pixels_per_obj):
        mask = (semantic_obs == obj_id).astype(np.int32)
        avg_depth = np.sum(depth_obs * mask) / np.sum(mask)
        avg_depths.append(avg_depth)
        if obj_id != target_obj and avg_depth > max_depth:
            depth_filtered.append(obj_id)
            continue
        if total_pixels / area > threshold:
            objects.append(obj_id)

    # print("Post depth filtering: {}/{} - {}".format(len(depth_filtered), len(obj_ids), avg_depths))

    return objects, avg_depths, obj_ids, depth_filtered


def plot_area(
    scene: str,
    obj: SemanticObject,
    points: List[ndarray],
    points_iou: List[ndarray],
) -> None:
    max_coord = 1000
    image = np.zeros((max_coord, max_coord, 3), dtype=np.uint8)

    def mark_points(points, color):
        int_points = [
            (int(p[0] * 10 + max_coord / 2), int(p[2] * 10 + max_coord / 2))
            for p in points
        ]
        for p in int_points:
            image[p[0], p[1]] = color

    def iou_points(points, color):
        int_points = [
            (
                int(p[0] * 10 + max_coord / 2),
                int(p[2] * 10 + max_coord / 2),
                iou,
            )
            for iou, p, _ in points
        ]
        for p in int_points:
            if p[2] == -1:
                image[p[0], p[1]] = NON_NAVIGABLE
            elif p[2] == -0.5:
                image[p[0], p[1]] = MAX_DIST
            else:
                color = int((p[2] + 1.0) * 255)
                image[p[0], p[1]] = [color, color, color]

    iou_points(points, POINT_COLOR)
    mark_points(points_iou, VIEW_POINT_COLOR)
    mark_points([obj.aabb.center], CENTER_POINT_COLOR)

    save_to = os.path.join(
        IMAGE_DIR, short_scene(scene), obj.id, "viewpoint_grid.png"
    )
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    imageio.imsave(save_to, image)


def draw_border(img: ndarray, thickness: int, color: ndarray) -> ndarray:
    vbar = np.ones((img.shape[0], thickness, 3)) * color
    img = np.concatenate([vbar, img, vbar], axis=1)
    hbar = np.ones((thickness, img.shape[1], 3)) * color
    img = np.concatenate([hbar, img, hbar], axis=0)
    return img


def save_hfov_test_imgs(
    obs: List[Dict[str, ndarray]],
    hfovs: List[float],
    save_to: Optional[str] = None,
    save_gif: bool = True,
    save_imgs: bool = True,
) -> None:
    """Write HFOV on each image and save to disk"""
    if save_to is None:
        save_to = os.path.join(IMAGE_DIR, "hfov_test")

    os.makedirs(save_to, exist_ok=True)

    loc = (50, 50)
    lt = cv2.LINE_AA

    images = []
    for o, hfov in zip(obs, hfovs):
        txt = f"HFOV: {hfov}"
        img = obs_to_frame(o)
        img = cv2.putText(img, txt, loc, FONT, 1, BLACK, 6, lt)
        img = cv2.putText(img, txt, loc, FONT, 1, WHITE, 3, lt)
        if save_gif:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if save_imgs:
            path = os.path.join(save_to, "src", f"hfov_{hfov}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, img)

    if save_gif:
        path = os.path.join(save_to, "hfov.gif")
        imageio.mimsave(path, images)


def log_text(verbose, scene, obj, obj_cat, text):
    if not verbose:
        return

    path = os.path.join(IMAGE_DIR, short_scene(scene), obj.id, "stats.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(f"Object ID: {obj.id}\n")
            f.write(f"Object Category: {obj_cat}\n")
    else:
        with open(path, "a") as f:
            f.write(f"{text}\n")


def clear_log(verbose, scene):
    path = os.path.join(IMAGE_DIR, short_scene(scene))
    if os.path.exists(path) and verbose:
        shutil.rmtree(path)
