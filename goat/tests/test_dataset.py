import argparse

from habitat import get_config

from goat.dataset.goat_dataset import GoatDatasetV1


def main(config_path):
    config = get_config(config_path)
    print(config.keys())

    dataset = GoatDatasetV1(config.habitat.dataset)
    print("Episodes: ", len(dataset.episodes))

    for i, episode in enumerate(dataset.episodes):
        print("Goals: {}".format(len(episode.goals), len(episode.tasks)))
        for idx, goal in enumerate(dataset.episodes[0].goals):
            print("[Goal id {}] Keys: {}".format(idx, len(goal)))

        if i == 100:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/tasks/goat_stretch_hm3d.yaml",
    )
    args = parser.parse_args()
    main(args.config_path)
