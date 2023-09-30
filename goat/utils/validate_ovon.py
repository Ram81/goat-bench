from goat.utils.utils import count_episodes, load_json


def main(path):
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
    main("data/datasets/ovon/hm3d/v2/train/content/")
