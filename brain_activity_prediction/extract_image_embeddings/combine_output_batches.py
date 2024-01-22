import pickle
import sys


def main():
    batch_paths = sorted(
        BATCH_FILES,
        key=lambda i: int(i.split("_")[-1].split(".")[0]),
    )

    hidden_layers = {}

    for batch_filepath in batch_paths:
        with open(batch_filepath, "rb") as batch_file:
            batches = pickle.load(batch_file)

        for batch in batches:
            keys = set([k for k in batch.keys() if k != "image_ids"])

            for image_num, image_id in enumerate(batch["image_ids"]):
                for key in keys:
                    if key not in hidden_layers:
                        hidden_layers[key] = {}

                    hs = tuple(hs[image_num] for hs in batch[key])
                    hidden_layers[key][image_id] = hs

            del batch

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(hidden_layers, f)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: <OUTPUT_PATH> <INPUT_PATHS...>")
        exit(1)

    OUTPUT_PATH = sys.argv[1]
    BATCH_FILES = sys.argv[2:]

    main()
