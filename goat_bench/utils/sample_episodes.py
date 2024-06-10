import argparse
import glob
import os
import os.path as osp
import random
import shutil

import tqdm

from goat_bench.utils.utils import load_dataset, write_dataset


def clean_instruction(instruction):
    uuid = instruction.lower()
    first_3_words = [
        "prefix: instruction: go",
        "instruction: find the",
        "instruction: go to",
        "api_failure",
        "instruction: locate the",
    ]
    for prefix in first_3_words:
        uuid = uuid.replace(prefix, "")
        uuid = uuid.replace("\n", " ")
    uuid = uuid.strip()
    return uuid


def main(input_path, output_path, max_episodes):
    files = glob.glob(osp.join(input_path, "*.json.gz"))
    num_gz_files = len(files)

    os.makedirs(output_path, exist_ok=True)

    num_added = 0
    for idx, file in enumerate(tqdm.tqdm(files)):
        dataset = load_dataset(file)
        random.shuffle(dataset["episodes"])

        episodes = []
        for episode in dataset["episodes"]:
            if episode.get("instructions") is not None:
                instruction = clean_instruction(episode["instructions"][0])
                if len(instruction) == 0:
                    continue
                episodes.append(episode)
            else:
                episodes.append(episode)

        num_left = max_episodes - num_added
        num_gz_remaining = num_gz_files - idx
        num_needed = min(num_left / num_gz_remaining, len(episodes))

        sampled_episodes = random.sample(episodes, int(num_needed))
        num_added += len(sampled_episodes)

        dataset["episodes"] = sampled_episodes

        output_file = osp.join(output_path, osp.basename(file))
        print(f"Copied {len(sampled_episodes)} episodes to {output_file}!")
        write_dataset(dataset, output_file)

    print(f"Added {num_added} episodes in total!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        help="Path to episode dir containing content/",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to episode dir containing content/",
    )
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.max_episodes)
