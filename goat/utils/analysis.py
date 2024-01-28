import argparse
import glob
import math
import os
from collections import defaultdict

from tqdm import tqdm

from goat.utils.utils import (
    count_episodes,
    load_dataset,
    load_json,
    load_pickle,
    write_json,
)


def validate_goat(path):
    files = glob.glob(os.path.join(path, "*.json.gz"))

    categories = defaultdict(int)
    iin_instances = 0
    lang_instances = 0
    object_goals = defaultdict(int)
    language_goals = defaultdict(int)
    for file in tqdm(files):
        dataset = load_dataset(file)

        goals = dataset["goals"]

        for episode in dataset["episodes"]:
            episode["goals"] = []
            for goal in episode["tasks"]:
                goal_type = goal[1]
                goal_category = goal[0]
                goal_inst_id = goal[2]

                dset_same_cat_goals = [
                    x
                    for x in goals.values()
                    if x[0]["object_category"] == goal_category
                ]
                children_categories = dset_same_cat_goals[0][0][
                    "children_object_categories"
                ]
                for child_category in children_categories:
                    goal_key = "{}_{}".format(
                        episode["scene_id"].split("/")[-1],
                        child_category,
                    )
                    if goal_key not in goals:
                        continue
                    dset_same_cat_goals[0].extend(goals[goal_key])

                assert (
                    len(dset_same_cat_goals) == 1
                ), f"more than 1 goal categories for {goal_category}"

                if goal_type == "object":
                    episode["goals"].append(dset_same_cat_goals[0])
                else:
                    goal_inst = [
                        x
                        for x in dset_same_cat_goals[0]
                        if x["object_id"] == goal_inst_id
                    ]
                    episode["goals"].append(goal_inst)

                if goal_type == "object":
                    object_goals[goal_category] += 1
                elif goal_type == "description":
                    language_goals[goal_inst[0]["lang_desc"].lower()] += 1

    write_json(object_goals, os.path.join(path, "goat_object_goals.json"))
    write_json(language_goals, os.path.join(path, "goat_language_goals.json"))

    categories = list(object_goals.keys())
    print(categories)
    instructions = list(language_goals.keys())
    # print(instructions[0:50], len(instructions[0:50]))

    synms = load_json("data/datasets/goat/language_goals_noise.json")
    # gpt3_synms = list(
    #     load_json("data/datasets/goat/language_goals.json").values()
    # )
    # print(len(gpt3_synms), 354 - 90)

    # synms = {}
    # for i in range(0, len(gpt3_synms)):
    #     synms[instructions[i]] = gpt3_synms[i]

    # write_json(synms, "data/datasets/goat/language_goals_noise.json")
    diff = set(instructions).symmetric_difference(set(synms.keys()))
    print(len(synms), len(diff))

    print("Categories: {}".format(len(object_goals)))
    print("LanguageNav instances:{}".format(len(language_goals)))


def goat_analysis(path):
    episode_metrics = load_json(path)

    spl_by_subtask_idx = defaultdict(float)
    success_by_subtask_idx = defaultdict(float)
    success_by_subtask_type = defaultdict(float)
    spl_by_subtask_type = defaultdict(float)
    count_by_subtask_type = defaultdict(float)

    print(episode_metrics[0].keys(), episode_metrics[0]["subtasks"])

    lang_avg_len = 0
    obj_avg_len = 0
    lang_n_eps = 0
    obj_n_eps = 0
    for metrics in episode_metrics:
        for i in range(0, 5):
            if i < 4:
                spl_by_subtask_idx[i + 1] += metrics["spl_by_subtaskl"][i]
                success_by_subtask_idx[i + 1] += metrics["success_by_subtask"][
                    i
                ]
            else:
                rem_spl = [
                    metrics["spl_by_subtaskl"][j]
                    for j in range(i, len(metrics["spl_by_subtaskl"]))
                ]
                spl_by_subtask_idx[i + 1] += sum(rem_spl) / len(rem_spl)

                rem_success = [
                    metrics["success_by_subtask"][j]
                    for j in range(i, len(metrics["success_by_subtask"]))
                ]
                success_by_subtask_idx[i + 1] += sum(rem_success) / len(
                    rem_success
                )
            if metrics["subtasks"][i][1] == "description":
                lang_avg_len += metrics["spl_by_subtaskl"][i]
                lang_n_eps += metrics["success_by_subtask"][i]

            if metrics["subtasks"][i][1] == "object":
                obj_avg_len += metrics["spl_by_subtaskl"][i]
                obj_n_eps += metrics["success_by_subtask"][i]

        for subtask_idx, subtask_meta in enumerate(metrics["subtasks"]):
            success_by_subtask_type[subtask_meta[1]] += metrics[
                "success_by_subtask"
            ][subtask_idx]
            spl_by_subtask_type[subtask_meta[1]] += metrics["spl_by_subtaskl"][
                subtask_idx
            ]
            count_by_subtask_type[subtask_meta[1]] += 1

    for k, v in spl_by_subtask_idx.items():
        spl_by_subtask_idx[k] = round(v / len(episode_metrics), 3) * 100
        success_by_subtask_idx[k] = (
            round(success_by_subtask_idx[k] / len(episode_metrics), 3) * 100
        )

    for k, v in success_by_subtask_type.items():
        success_by_subtask_type[k] = (
            round(success_by_subtask_type[k] / count_by_subtask_type[k], 3)
            * 100
        )
        spl_by_subtask_type[k] = (
            round(spl_by_subtask_type[k] / count_by_subtask_type[k], 3) * 100
        )

    print()
    print("SPL by subtask number", spl_by_subtask_idx)
    print("Success by subtask number", success_by_subtask_idx)
    print("Success by subtask type", success_by_subtask_type)
    print("SPL by subtask type", spl_by_subtask_type)

    print("\nAvg lang desc len: {}".format(lang_avg_len / lang_n_eps))
    print("Avg von desc len: {}".format(obj_avg_len / obj_n_eps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the dataset",
    )

    args = parser.parse_args()

    # validate_goat(args.path)
    goat_analysis(args.path)
