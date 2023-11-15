import argparse
import glob
import os

from goat.utils.utils import write_json


def main(path, output_path):
    splits = ["val_seen", "val_unseen_easy", "val_unseen_hard"]

    records = []
    for split in splits:
        files = glob.glob(os.path.join(path, split, "*/*png"))

        for file in files:
            records.append(
                {
                    "img_path": file.replace("data/mturk/", ""),
                    "caption": "",
                    "split": split,
                    "object_category": file.split("/")[-1]
                    .split("_")[2]
                    .split(".")[0],
                }
            )

    write_json(records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    main(args.path, args.output_path)
