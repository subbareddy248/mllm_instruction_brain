import argparse
from collections import OrderedDict
import pathlib
import pickle


from datasets import Dataset
import torch
import transformers
from tqdm.auto import tqdm

import nsd_dataset.mind_eye_nsd_utils as menutils

MODEL_ID = "Salesforce/instructblip-vicuna-7b"
CONFIG_CLASS = transformers.InstructBlipConfig
MODEL_CLASS = transformers.InstructBlipForConditionalGeneration
PROCESSOR_CLASS = transformers.InstructBlipProcessor

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
    model_config.vision_config.output_hidden_states = True
    model_config.qformer_config.output_hidden_states = True

    model = MODEL_CLASS.from_pretrained(
        MODEL_ID,
        config=model_config,
        cache_dir=HUGGINGFACE_CACHE_DIR,
        device_map="auto",
        max_memory={
            0: "7GB",
            GPU_ID: "7GB",
        },
        low_cpu_mem_usage=True,
        offload_folder=HUGGINGFACE_CACHE_DIR.joinpath("offload", MODEL_NAME),
    )

    image_ids, images = menutils.get_subject_images(BASE_DIR, SUBJECT)

    def data_generator():
        for image_id, image in zip(image_ids, images):
            yield {"id": image_id, "image": image}

    dataset = Dataset.from_generator(data_generator)

    batches = batchify(dataset, n=BATCH_SIZE)
    total_batches = len(dataset) // BATCH_SIZE

    model.eval()

    BUFFER = []

    with torch.inference_mode():
        for batch_num, batch in tqdm(enumerate(batches), total=total_batches):
            images = torch.tensor(batch["image"])
            text = [PROMPT] * BATCH_SIZE

            inputs = processor(images=images, text=text, return_tensors="pt")

            outputs = model(**inputs)
            outputs = to_cpu(outputs)

            if TEST_RUN:
                print("outputs =", outputs)
                exit(0)

            vision_outputs = outputs["vision_outputs"]
            qformer_outputs = outputs["qformer_outputs"]

            vision_hidden_states = tuple(torch.mean(hs, dim=1).numpy() for hs in vision_outputs["hidden_states"])
            qformer_hidden_states = tuple(torch.mean(hs, dim=1).numpy() for hs in qformer_outputs["hidden_states"])

            BUFFER.append(
                {
                    "image_ids": batch["id"],
                    "vision_hidden_states": vision_hidden_states,
                    "qformer_hidden_states": qformer_hidden_states,
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

    args = parser.parse_args()

    BATCH_SIZE: int = args.batch_size
    BASE_DIR: pathlib.Path = args.base_dir
    SUBJECT: int = args.subject
    TEST_RUN: bool = args.test_run
    PROMPT: str = args.prompt
    GPU_ID: int = args.gpu_id

    HUGGINGFACE_CACHE_DIR = BASE_DIR.joinpath(".huggingface_cache")
    OUTPUT_DIR = BASE_DIR.joinpath("image_embeddings", MODEL_NAME, f"subject_0{SUBJECT}")
    MODEL_CHECKPOINTS_DIR = BASE_DIR.joinpath("cached_models", MODEL_NAME)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    main()
