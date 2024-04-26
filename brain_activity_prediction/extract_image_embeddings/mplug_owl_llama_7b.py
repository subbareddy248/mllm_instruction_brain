import argparse
from collections import OrderedDict
import glob
import pathlib
import pickle
import PIL

from prompts import ALL_PROMPTS

from datasets import Dataset
import numpy as np
import torch

import nsd_dataset.mind_eye_nsd_utils as menutils

from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration, MplugOwlConfig
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from mplug_owl.processing_mplug_owl import MplugOwlProcessor, MplugOwlImageProcessor


MODEL_ID = "MAGAer13/mplug-owl-llama-7b"
MODEL_CLASS = MplugOwlForConditionalGeneration

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
    image_processor = MplugOwlImageProcessor.from_pretrained(MODEL_ID) # , cache_dir=HUGGINGFACE_CACHE_DIR)
    tokenizer = MplugOwlTokenizer.from_pretrained(MODEL_ID) # , cache_dir=HUGGINGFACE_CACHE_DIR)
    processor = MplugOwlProcessor(image_processor, tokenizer)

    model_config = MplugOwlConfig.from_pretrained(MODEL_ID)
    model_config.text_config.output_hidden_states = True
    model_config.vision_config.output_hidden_states = True

    model = MODEL_CLASS.from_pretrained(
        MODEL_ID,
        # cache_dir=HUGGINGFACE_CACHE_DIR,
    ).to(GPU_DEVICE)

    def data_generator():
        image_ids, images = menutils.get_subject_images(BASE_DIR, SUBJECT)

        for image_id, image in zip(image_ids, images):
            yield {"id": image_id, "image": image}

    dataset = Dataset.from_generator(data_generator) # , cache_dir=HUGGINGFACE_CACHE_DIR.joinpath("datasets"))

    batches = batchify(dataset, n=BATCH_SIZE)
    total_batches = len(dataset) // BATCH_SIZE

    model.eval()

    BUFFER = []

    with torch.inference_mode():
        batch_iter = enumerate(batches)
        if TO_USE_TELEGRAM:
            batch_iter = tqdm(
                batch_iter,
                desc=f"[mPlugOwl] (Prompt_{args.prompt_number}) Subject 0{SUBJECT}",
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

        done_ids = []
        batch_files = glob.glob(str(pathlib.Path(OUTPUT_DIR).joinpath("batch_*.pkl")))
        for batch_file in batch_files:
            with open(batch_file, "rb") as f:
                batches = pickle.load(f)
            for batch in batches:
                done_ids.extend(batch["image_ids"])
        done_ids = set(done_ids)

        for batch_num, batch in batch_iter:
            if all(image_id in done_ids for image_id in batch["id"]):
                continue

            images = [PIL.Image.fromarray(np.array(image).astype("uint8"), "RGB") for image in batch["image"]]
            text = [PROMPT] * BATCH_SIZE

            inputs = processor(images=images, text=text, return_tensors="pt")
            for k, v in inputs:
                inputs[k] = v.to(GPU_DEVICE)

            non_padding_mask = ((inputs.input_ids != tokenizer.pad_token_id)[:, :-1]).to(GPU_DEVICE)
            non_media_mask = torch.ones_like(non_padding_mask).to(GPU_DEVICE)
            prompt_mask = torch.zeros_like(non_padding_mask).to(GPU_DEVICE)

            outputs = model(
                **inputs,
                output_hidden_states=True,
                num_images=torch.tensor([len(images)]),
                non_padding_mask=non_padding_mask,
                non_media_mask=non_media_mask,
                prompt_mask=prompt_mask,
                labels=inputs.input_ids,
            )
            outputs = to_cpu(outputs)

            if TEST_RUN:
                print(f"{outputs.keys() = }")
                with open("temp_outputs.pkl", "wb") as f:
                    pickle.dump(outputs, f)
                exit(0)

            vision_outputs = outputs["vision_outputs"]
            language_outputs = outputs["language_outputs"]

            vision_hidden_states = tuple(torch.mean(hs, dim=1).numpy() for hs in vision_outputs["hidden_states"])
            language_hidden_states = tuple(torch.mean(hs, dim=1).numpy() for hs in language_outputs["hidden_states"])

            BUFFER.append(
                {
                    "image_ids": batch["id"],
                    "vision_hidden_states": vision_hidden_states,
                    "language_hidden_states": language_hidden_states,
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
    # parser.add_argument(
    #     "-p",
    #     "--prompt",
    #     required=False,
    #     default=(
    #         "The following is a conversation between a curious human and AI assistant. "
    #         "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    #         "Human: <image>\n"
    #         "Human: What's the content of the image?\n"
    #         "AI:"
    #     ),
    #     type=str,
    #     help="The prompt that will be passed into the model along with the images",
    # )
    parser.add_argument(
        "-p",
        "--prompt-number",
        required=True,
        type=int,
        help="The number of the prompt that will be passed into the model along with the images",
    )
    parser.add_argument(
        "-g",
        "--gpu-id",
        required=True,
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
    PROMPT_NUMBER: int = args.prompt_number
    GPU_ID: int = args.gpu_id
    TELEGRAM_BOT_TOKEN: str = args.telegram_bot_token
    TELEGRAM_CHAT_ID: int = args.telegram_chat_id
    TO_USE_TELEGRAM: bool = TELEGRAM_BOT_TOKEN != "" and TELEGRAM_CHAT_ID != 0

    if TO_USE_TELEGRAM:
        from tqdm.contrib.telegram import tqdm
    else:
        from tqdm.auto import tqdm

    GPU_DEVICE: str = f"cuda:{GPU_ID}"

    PROMPT = (
        "The following is a conversation between a curious human and AI assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        "Human: <image>\n"
        f"Human: {ALL_PROMPTS[PROMPT_NUMBER]}\n"
        "AI:"
    )

    # HUGGINGFACE_CACHE_DIR = BASE_DIR.joinpath(".huggingface_cache")
    OUTPUT_DIR = BASE_DIR.joinpath("image_embeddings", f"prompt_{args.prompt_number}", MODEL_NAME, f"subject_0{SUBJECT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    main()
