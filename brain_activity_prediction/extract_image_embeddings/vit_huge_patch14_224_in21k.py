import argparse
from collections import OrderedDict
import pathlib
import pickle


from datasets import Dataset
import torch
import transformers

import nsd_dataset.mind_eye_nsd_utils as menutils

MODEL_ID = "google/vit-huge-patch14-224-in21k"
CONFIG_CLASS = transformers.ViTConfig
MODEL_CLASS = transformers.ViTModel
PROCESSOR_CLASS = transformers.ViTImageProcessor

MODEL_NAME = MODEL_ID.replace("/", "_").replace(" ", "_")


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.cpu()
    elif isinstance(value, tuple):
        return tuple(to_cpu(v) for v in value)
    elif isinstance(value, list):
        return list(to_cpu(v) for v in value)
    elif isinstance(value, set):
        return set(list(to_cpu(v) for v in value))
    elif isinstance(value, dict) or isinstance(value, OrderedDict):
        return {k: to_cpu(v) for k, v in value.items()}
    else:
        print("unknown type:", value)


def batchify(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def main():
    processor = PROCESSOR_CLASS.from_pretrained(MODEL_ID, cache_dir=HUGGINGFACE_CACHE_DIR)

    model_config = CONFIG_CLASS.from_pretrained(MODEL_ID)
    model_config.output_hidden_states = True

    model = MODEL_CLASS.from_pretrained(
        MODEL_ID,
        config=model_config,
        cache_dir=HUGGINGFACE_CACHE_DIR,
        device_map="auto",
        max_memory={
            GPU_ID: "7GB",
        },
        low_cpu_mem_usage=True,
        offload_folder=HUGGINGFACE_CACHE_DIR.joinpath("offload", MODEL_NAME),
    )

    def data_generator():
        image_ids, images = menutils.get_subject_images(BASE_DIR, SUBJECT)

        for image_id, image in zip(image_ids, images):
            yield {"id": image_id, "image": image}

    dataset = Dataset.from_generator(data_generator, cache_dir=HUGGINGFACE_CACHE_DIR.joinpath("datasets"))

    batches = batchify(dataset, n=BATCH_SIZE)
    total_batches = len(dataset) // BATCH_SIZE

    model.eval()

    BUFFER = []

    with torch.inference_mode():

        batch_iter = enumerate(batches)
        if TO_USE_TELEGRAM:
            batch_iter = tqdm(
                batch_iter,
                desc=f"[ViTHuge] Subject 0{SUBJECT}",
                mininterval=10,
                maxinterval=20,
                total=total_batches,
                token=TELEGRAM_BOT_TOKEN,
                chat_id=TELEGRAM_CHAT_ID,
            )
        else:
            batch_iter = tqdm(
                batch_iter,
                total=total_batches,
            )

        for batch_num, batch in batch_iter:
            images = torch.tensor(batch["image"])
            text = [PROMPT] * BATCH_SIZE

            inputs = processor(images=images, text=text, return_tensors="pt")

            outputs = model(**inputs, output_hidden_states=True)
            outputs = to_cpu(outputs)

            if TEST_RUN:
                print(f"{outputs['hidden_states'][0].shape =}")
                exit(0)

            hidden_states = outputs["hidden_states"]

            hidden_states = tuple(torch.mean(hs, dim=1).numpy() for hs in hidden_states)

            BUFFER.append(
                {
                    "image_ids": batch["id"],
                    "hidden_states": hidden_states,
                }
            )

            if len(BUFFER) == 100:
                batch_file = pathlib.Path(OUTPUT_DIR).joinpath(f"batch_{batch_num+1}.pkl")
                with open(batch_file, "wb") as f:
                    pickle.dump(BUFFER, f)
                del BUFFER
                BUFFER = []

        if len(BUFFER) > 0:
            batch_file = pathlib.Path(OUTPUT_DIR).joinpath(f"batch_{batch_num+1}.pkl")
            with open(batch_file, "wb") as f:
                pickle.dump(BUFFER, f)
            del BUFFER
            BUFFER = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Embeddings Extractor",
        description=f"This program will extract image embeddings from the model {MODEL_ID}",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        required=True,
        default=1,
        type=int,
        help="The number of images which will be passed into the model together",
    )
    parser.add_argument(
        "-d",
        "--base-dir",
        required=False,
        default=pathlib.Path("/tmp/akshett.jindal"),
        type=pathlib.Path,
        help="The path to the directory where all the models, inputs and the outputs will be cached and loaded from",
    )
    parser.add_argument(
        "-s",
        "--subject",
        required=True,
        type=int,
        choices=set([1, 2, 5, 7]),
        help="The subject number from the NSD dataset whose images' embeddings are to be extracted",
    )
    parser.add_argument(
        "-t",
        "--test-run",
        required=False,
        default=False,
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Enable test-run to just output the outputs for the first batch and exit",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        required=False,
        default="Describe the image.",
        type=str,
        help="The prompt that will be passed into the model along with the images",
    )
    parser.add_argument(
        "-g",
        "--gpu-id",
        required=False,
        default=0,
        type=int,
        help="The CUDA GPU id on which to run inference",
    )
    parser.add_argument(
        "--telegram-bot-token",
        required=False,
        default="",
        type=str,
        help="Telegram Bot token to use for tqdm",
    )
    parser.add_argument(
        "--telegram-chat-id",
        required=False,
        default=0,
        type=int,
        help="Telegram Chat ID to send tqdm updates to",
    )

    args = parser.parse_args()

    BATCH_SIZE: int = args.batch_size
    BASE_DIR: pathlib.Path = args.base_dir
    SUBJECT: int = args.subject
    TEST_RUN: bool = args.test_run
    PROMPT: str = args.prompt
    GPU_ID: int = args.gpu_id
    TELEGRAM_BOT_TOKEN: str = args.telegram_bot_token
    TELEGRAM_CHAT_ID: int = args.telegram_chat_id
    TO_USE_TELEGRAM: bool = TELEGRAM_BOT_TOKEN != "" and TELEGRAM_CHAT_ID != 0

    if TO_USE_TELEGRAM:
        from tqdm.contrib.telegram import tqdm
    else:
        from tqdm.auto import tqdm

    HUGGINGFACE_CACHE_DIR = BASE_DIR.joinpath(".huggingface_cache")
    OUTPUT_DIR = BASE_DIR.joinpath("image_embeddings", MODEL_NAME, f"subject_0{SUBJECT}")
    MODEL_CHECKPOINTS_DIR = BASE_DIR.joinpath("cached_models", MODEL_NAME)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    main()
