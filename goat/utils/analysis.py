import argparse
import glob
import os
from collections import defaultdict

from goat.utils.utils import (
    count_episodes,
    load_dataset,
    load_pickle,
    write_json,
)


def validate_goat(path):
    files = glob.glob(os.path.join(path, "*.json.gz"))
    total = 0
    categories = defaultdict(int)
    iin_instances = 0
    lang_instances = 0
    for file in files:
        dataset = load_dataset(file)

        for g_key, goals in dataset["goals"].items():
            category = g_key.split("_")[1]
            categories[category] += 1
            for goal in goals:
                if goal.get("lang_desc") is not None:
                    lang_instances += 1
                if goal.get("image_goals") is not None:
                    iin_instances += 1

    print("Categories: {}".format(categories))
    print("L: {}, I:{}".format(iin_instances, lang_instances))
    print("L: {}".format(len(categories)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the dataset",
    )

    args = parser.parse_args()

    validate_goat(args.path)
