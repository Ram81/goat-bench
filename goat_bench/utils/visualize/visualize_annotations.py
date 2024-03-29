import glob
import os

import numpy as np
from habitat.utils.visualizations.utils import append_text_to_image
from tqdm import tqdm

from goat_bench.utils.utils import load_image, load_json, save_image


def visualize_annotations(annotations, output_path):
    obs_path = list(annotations.values())[0]["observation"]
    obs_path = os.path.dirname(obs_path)

    missing_annotations = 0
    files = glob.glob(obs_path + "/*png")
    for file in files:
        object_id = file.split("/")[-1].split(".")[0]

        if annotations.get(object_id) is None:
            missing_annotations += 1
            continue

        instruction = annotations[object_id]["instructions"]["@1"]
        img = np.array(load_image(file))
        img = append_text_to_image(img, instruction)
        save_image(img, os.path.join(output_path, "{}.png".format(object_id)))
    print("Missing annotations: {}".format(missing_annotations))


def visualize(path, output_path):
    files = glob.glob(path + "/*_annotated.json")

    for file in tqdm(files):
        opath = os.path.join(output_path, file.split("/")[-1].split("_")[0])
        os.makedirs(opath, exist_ok=True)
        print("Out path: {}".format(opath))

        annotations = load_json(file)
        visualize_annotations(annotations, opath)


def validate(path):
    files = glob.glob(path + "/*annotated.json")

    for file in files:
        scene = file.split("/")[-1].split("_")[0]
        print("Scene: {}".format(scene))

        viewpoints = load_json(file.replace("_annotated", ""))
        annotated_viewpoints = load_json(file)
        failures = 0
        for uuid, annotation in annotated_viewpoints.items():
            if "Failure" in annotation["instructions"]["@1"]:
                failures += 1

        uuids = set(list(annotated_viewpoints.keys()))
        vp_uuids = set(list(viewpoints.keys()))
        if len(vp_uuids.difference(uuids)) != 0:
            print("Incomplete anotations for scene {}!".format(scene))

        print(
            "Total failures: {}/{}".format(failures, len(annotated_viewpoints))
        )


if __name__ == "__main__":
    output_path = "data/visualizations/language_nav_v5/"
    os.makedirs(output_path, exist_ok=True)
    visualize(
        "data/datasets/languagenav/hm3d/v5_final/train/content/",
        output_path,
    )
