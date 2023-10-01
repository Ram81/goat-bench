import argparse
import glob
import os

from goat.utils.utils import count_episodes, load_dataset, load_json, write_json


def save_instances(path, output_path):
    files = glob.glob(os.path.join(path, "*.json.gz"))

    data = {}
    total_instances = 0
    for file in files:
        dataset = load_dataset(file)
        scene_id = file.split("/")[-1].split(".")[0]

        data[scene_id] = {}
        for category, instances in dataset["goals_by_category"].items():
            for instance in instances:
                data[scene_id][instance["object_id"]] = {
                    "category": instance["object_category"],
                    "position": instance["position"],
                }
                total_instances += 1

    print(
        "Total instances: {}, Split: {}".format(
            total_instances, path.split("/")[-2]
        )
    )
    write_json(data, output_path)


def save_ovon_instances(path, output_path):
    splits = ["train", "val_seen", "val_unseen_easy", "val_unseen_hard"]

    for split in splits:
        split_path = os.path.join(path, split, "content")
        save_instances(
            split_path,
            os.path.join(output_path, split + "_object_instances.json"),
        )


def validate_lnav(path):
    _, train_categories = count_episodes(os.path.join(path, "train", "content"))

    _, val_categories = count_episodes(os.path.join(path, "val", "content"))
    val_seen_diff = set(val_categories.keys()) - set(train_categories.keys())
    print("Total train categories: {}".format(len(train_categories)))

    print(
        "Total diff val seen categories: {} - {}".format(
            len(val_seen_diff), len(val_categories)
        )
    )

    splits = ["train", "val_seen", "val_unseen_easy", "val_unseen_hard"]
    categories_by_split = {}
    ovon_path = "data/datasets/ovon/hm3d/v5_final/"
    for split in splits:
        split_path = os.path.join(ovon_path, split, "content")
        files = glob.glob(os.path.join(split_path, "*.json.gz"))
        categories = []
        for file in files:
            dataset = load_dataset(file)
            for cat, instances in dataset["goals_by_category"].items():
                categories.append(cat.split("_")[1])
        categories_by_split[split] = set(categories)

    lnav_train_categories = list(train_categories.keys())
    lnav_val_categories = list(val_categories.keys())

    train_diff = set(lnav_train_categories) - categories_by_split["train"]
    val_diff = (
        set(lnav_val_categories)
        - categories_by_split["val_seen"]
        - categories_by_split["val_unseen_easy"]
        - categories_by_split["val_unseen_hard"]
    )
    ovon_diff = categories_by_split["train"] - set(lnav_train_categories)
    all_val_categories = (
        list(categories_by_split["val_seen"])
        + list(categories_by_split["val_unseen_easy"])
        + list(categories_by_split["val_unseen_hard"])
    )
    ovon_val_diff = set(all_val_categories) - set(lnav_val_categories)
    ovon_vue_diff = categories_by_split["val_unseen_easy"] - set(
        lnav_val_categories
    )
    ovon_vuh_diff = categories_by_split["val_unseen_hard"] - set(
        lnav_val_categories
    )
    print(
        "Train categories in OVON: {} - {} - {}".format(
            len(train_diff), len(ovon_diff), len(lnav_train_categories)
        )
    )
    print(
        "Val categories in OVON: {} - {} - {}".format(
            len(val_diff), len(ovon_val_diff), len(lnav_val_categories)
        )
    )
    print(
        "Val unseen easy categories in OVON: {} - {}".format(
            len(ovon_vue_diff), len(lnav_val_categories)
        )
    )
    print(
        "Val unseen hard categories in OVON: {} - {}".format(
            len(ovon_vuh_diff), len(lnav_val_categories)
        )
    )


def validate_ovon(path):
    counts, train_categories = count_episodes(path)

    val_path = "data/datasets/ovon/hm3d/v3_shuffled_cleaned/{}/content/"
    counts, val_seen_categories = count_episodes(val_path.format("val_seen"))
    counts, val_unseen_easy_categories = count_episodes(
        val_path.format("val_unseen_easy")
    )
    counts, val_unseen_hard_categories = count_episodes(
        val_path.format("val_unseen_hard")
    )

    val_seen_diff = set(val_seen_categories.keys()) - set(
        train_categories.keys()
    )
    val_unseen_easy_diff = set(val_unseen_easy_categories.keys()) - set(
        train_categories.keys()
    )
    val_unseen_hard_diff = set(val_unseen_hard_categories.keys()) - set(
        train_categories.keys()
    )

    print("Total train categories: {}".format(len(train_categories)))

    print(
        "Total diff val seen categories: {} - {}".format(
            len(val_seen_diff), len(val_seen_categories)
        )
    )
    print(
        "Total diff val unseen easy categories: {} - {}".format(
            len(val_unseen_easy_diff), len(val_unseen_easy_categories)
        )
    )
    print(
        "Total diff val unseen hard categories: {} - {}".format(
            len(val_unseen_hard_diff), len(val_unseen_hard_categories)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        default="",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--save-instances",
        action="store_true",
        dest="save_instances",
    )

    args = parser.parse_args()

    if args.save_instances:
        save_ovon_instances(args.path, args.output_path)
    elif "languagenav" in args.path:
        validate_lnav(args.path)
    else:
        validate_ovon("data/datasets/ovon/hm3d/v2/train/content/")
