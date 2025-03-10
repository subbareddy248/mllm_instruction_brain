import argparse
import glob
from himalaya.backend import set_backend
from npp import zscore
from himalaya.ridge import RidgeCV
from nsd_dataset import mind_eye_nsd_utils as menutils
import numpy
import pathlib
import pickle
from sklearn.pipeline import make_pipeline

LAYER_NUM_DEFAULT_VALUE = -10000


def train_model(
    hidden_states_name,
    hidden_states,
    trn_images,
    val_images,
    trn_voxel_data,
    val_voxel_data,
    alphas,
    backend,
):
    pipelines = []

    num_hidden_layers = 0
    if LAYER_NUM != LAYER_NUM_DEFAULT_VALUE:
        num_hidden_layers = 1
    else:
        for i in hidden_states.values():
            num_hidden_layers = len(i)
            break

    hidden_layers_iter = range(num_hidden_layers) if LAYER_NUM == LAYER_NUM_DEFAULT_VALUE else [LAYER_NUM]
    if TO_USE_TELEGRAM:
        hidden_layers_iter = tqdm(
            hidden_layers_iter,
            desc=f"[{MODEL_NAME}] (subj0{SUBJECT},{hidden_states_name})",
            mininterval=10,
            maxinterval=20,
            total=num_hidden_layers,
            token=TELEGRAM_BOT_TOKEN,
            chat_id=TELEGRAM_CHAT_ID,
        )
    else:
        hidden_layers_iter = tqdm(
            hidden_layers_iter,
            total=num_hidden_layers,
        )

    for layer_num in hidden_layers_iter:
        trn_hs = zscore(numpy.array([hidden_states[image_id + 1][layer_num] for image_id in trn_images]))
        val_hs = zscore(numpy.array([hidden_states[image_id + 1][layer_num] for image_id in val_images]))

        ridge_cv = RidgeCV(
            alphas=alphas,
            solver_params=dict(
                n_targets_batch=500,
                n_alphas_batch=10,
                n_targets_batch_refit=100,
            ),
        )
        pipeline = make_pipeline(ridge_cv)
        _ = pipeline.fit(trn_hs, trn_voxel_data)

        trn_scores = backend.to_numpy(pipeline.score(trn_hs, trn_voxel_data))
        val_scores = backend.to_numpy(pipeline.score(val_hs, val_voxel_data))

        trn_hs_pred = backend.to_numpy(pipeline.predict(trn_hs))
        val_hs_pred = backend.to_numpy(pipeline.predict(val_hs))

        trn_pearson_x, trn_pearson_y = trn_hs_pred.transpose(1, 0), trn_voxel_data.transpose(1, 0)
        val_pearson_x, val_pearson_y = val_hs_pred.transpose(1, 0), val_voxel_data.transpose(1, 0)

        trn_pearson_scores = []
        val_pearson_scores = []

        for voxel_num in range(trn_scores.shape[0]):
            trn_pearson = numpy.corrcoef(trn_pearson_x[voxel_num], trn_pearson_y[voxel_num])[0][1]
            val_pearson = numpy.corrcoef(val_pearson_x[voxel_num], val_pearson_y[voxel_num])[0][1]

            trn_pearson_scores.append(trn_pearson)
            val_pearson_scores.append(val_pearson)

        trn_pearson_scores = numpy.array(trn_pearson_scores)
        val_pearson_scores = numpy.array(val_pearson_scores)

        pipelines.append(
            {
                "pipeline": pipeline,
                "trn_scores": trn_scores,
                "val_scores": val_scores,
                "trn_pearson": trn_pearson_scores,
                "val_pearson": val_pearson_scores,
            }
        )

    return pipelines


def get_roi_scores(brain, voxels, scores, roi_masks):
    roi_scores = {}

    brain[:] = 0
    brain[voxels[0], voxels[1], voxels[2]] = scores
    for roi in roi_masks:
        if roi == "nsdgeneral":
            continue
        roi_mask = numpy.where(roi_masks[roi] == 1)
        roi_score = numpy.nanmean(brain[roi_mask[0], roi_mask[1], roi_mask[2]])
        roi_scores[roi] = roi_score

    return roi_scores


def main():

    batch_files = glob.glob(str(BASE_DIR.joinpath(
        "image_embeddings",
        f"prompt_{PROMPT_NUMBER}",
        MODEL_NAME,
        f"subject_{SUBJECT:02}",
        "*.pkl",
    )))

    HIDDEN_STATES = {}

    for batch_file in batch_files:
        with open(batch_file, "rb") as f:
            batches = pickle.load(f)

        for batch in batches:
            keys = set([k for k in batch.keys() if k not in ["image_ids", "generated_sentence"]])

            for image_num, image_id in enumerate(batch["image_ids"]):
                for key in keys:
                    if key not in HIDDEN_STATES:
                        HIDDEN_STATES[key] = {}

                    hs = tuple(hs[image_num] for hs in batch[key])
                    HIDDEN_STATES[key][image_id] = hs

            del batch

    print(f"{HIDDEN_STATES.keys() = }")

    (
        trial_order,
        session_data,
        (
            trn_stim_ordering,
            trn_voxel_data,
            val_stim_ordering,
            val_voxel_data,
        ),
    ) = menutils.get_split_data(BASE_DIR, SUBJECT, average_out_fmri=True)

    trn_voxel_data = zscore(trn_voxel_data)
    val_voxel_data = zscore(val_voxel_data)

    exp_design_file = menutils.load_exp_design_file(BASE_DIR)
    trial_images = exp_design_file["subjectim"]
    trn_images = trial_images[SUBJECT - 1, trn_stim_ordering] - 1
    val_images = trial_images[SUBJECT - 1, val_stim_ordering] - 1

    backend = set_backend("torch_cuda", on_error="warn")

    alphas = numpy.logspace(1, MAX_LOG_10_ALPHA, NUM_ALPHAS)

    roi_masks = menutils.load_roi_masks(BASE_DIR, SUBJECT)

    brain_mask = roi_masks["nsdgeneral"]
    voxels = numpy.where(brain_mask == 1)

    brain = numpy.zeros_like(brain_mask, dtype=numpy.float32)

    TRN_SCORES = {}
    VAL_SCORES = {}

    TRN_OUTPUT_FILE = OUTPUT_DIR.joinpath("training.pkl")
    VAL_OUTPUT_FILE = OUTPUT_DIR.joinpath("validation.pkl")
    VAL_PEARSONS_FILE = OUTPUT_DIR.joinpath("validation_pearsons.pkl")

    for hs_name in HIDDEN_STATES.keys():
        pipelines = train_model(
            hs_name,
            HIDDEN_STATES[hs_name],
            trn_images,
            val_images,
            trn_voxel_data,
            val_voxel_data,
            alphas,
            backend,
        )

        trn_scores = None
        val_scores = None

        trn_pearsons = None
        val_pearsons = None

        for stats in pipelines:
            trn_score = stats["trn_scores"]
            val_score = stats["val_scores"]

            trn_pearson = stats["trn_pearson"]
            val_pearson = stats["val_pearson"]

            if trn_scores is None:
                trn_scores = numpy.array([trn_score])
            else:
                trn_scores = numpy.concatenate((trn_scores, [trn_score]), axis=0)
            if val_scores is None:
                val_scores = numpy.array([val_score])
            else:
                val_scores = numpy.concatenate((val_scores, [val_score]), axis=0)

            if trn_pearsons is None:
                trn_pearsons = numpy.array([trn_pearson])
            else:
                trn_pearsons = numpy.concatenate((trn_pearsons, [trn_pearson]), axis=0)
            if val_pearsons is None:
                val_pearsons = numpy.array([val_pearson])
            else:
                val_pearsons = numpy.concatenate((val_pearsons, [val_pearson]), axis=0)

        with open(VAL_PEARSONS_FILE, "wb") as f:
            pickle.dump(val_pearsons, f)

        trn_scores = numpy.nanmean(trn_scores, axis=0)
        val_scores = numpy.nanmean(val_scores, axis=0)

        trn_pearsons = numpy.nanmean(trn_pearsons, axis=0)
        val_pearsons = numpy.nanmean(val_pearsons, axis=0)

        trn_roi_scores = get_roi_scores(brain, voxels, trn_scores, roi_masks)
        val_roi_scores = get_roi_scores(brain, voxels, val_scores, roi_masks)

        trn_roi_pearsons = get_roi_scores(brain, voxels, trn_pearsons, roi_masks)
        val_roi_pearsons = get_roi_scores(brain, voxels, val_pearsons, roi_masks)

        TRN_SCORES[hs_name] = {
            "r2": trn_roi_scores,
            "pearson": trn_roi_pearsons,
        }
        VAL_SCORES[hs_name] = {
            "r2": val_roi_scores,
            "pearson": val_roi_pearsons,
        }

    with open(TRN_OUTPUT_FILE, "wb") as f:
        pickle.dump(TRN_SCORES, f)
    with open(VAL_OUTPUT_FILE, "wb") as f:
        pickle.dump(VAL_SCORES, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Hidden States Aligner",
        description="This is a generic script that will train models to predict fMRI readings given model hidden states",
    )

    parser.add_argument(
        "-s",
        "--subject",
        required=True,
        type=int,
        choices=set([1, 2, 5, 7]),
        help="The subject number from the NSD dataset whose image embeddings are to be trained",
    )
    parser.add_argument(
        "-d",
        "--base-dir",
        required=False,
        type=pathlib.Path,
        # default=pathlib.Path("/tmp/akshett.jindal"),
        help="The path to the directory where all the models, inputs and outputs will be stored and loaded from",
    )
    parser.add_argument(
        "-m",
        "--model-id",
        type=str,
        required=True,
        help="The model id whose hidden state representations are to be used",
    )
    parser.add_argument(
        "-p",
        "--prompt-number",
        type=int,
        required=True,
        help="The prompt number to use for aligning",
    )
    parser.add_argument(
        "-l",
        "--layer-number",
        type=int,
        required=False,
        default=LAYER_NUM_DEFAULT_VALUE,
        help="The layer numbers to find the alignment for. It can be a number like 0, 1 or a negative number like -1 for the last layer. If not passed, then all the layers will be trained and average score will be extracted",
    )
    parser.add_argument(
        "--max-log-10-alpha",
        required=False,
        default=4,
        type=int,
        help="Maximum value of log10 alpha to consider",
    )
    parser.add_argument(
        "--num-alphas",
        required=False,
        default=60,
        type=int,
        help="Number of alpha values to sample",
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

    SUBJECT: int = args.subject
    BASE_DIR: pathlib.Path = args.base_dir
    MODEL_ID: str = args.model_id
    PROMPT_NUMBER: int = args.prompt_number
    MAX_LOG_10_ALPHA: int = args.max_log_10_alpha
    NUM_ALPHAS: int = args.num_alphas
    TELEGRAM_BOT_TOKEN: str = args.telegram_bot_token
    TELEGRAM_CHAT_ID: int = args.telegram_chat_id
    TO_USE_TELEGRAM: bool = TELEGRAM_BOT_TOKEN != "" and TELEGRAM_CHAT_ID != 0
    LAYER_NUM: int = args.layer_number

    MODEL_NAME = MODEL_ID.replace("/", "_").replace(" ", "_")

    if TO_USE_TELEGRAM:
        from tqdm.contrib.telegram import tqdm
    else:
        from tqdm.auto import tqdm

    OUTPUT_DIR = BASE_DIR.joinpath(
        "final_scores",
        f"prompt_{PROMPT_NUMBER}",
        MODEL_NAME,
        f"subj{SUBJECT:02}",
        f"layer_{'all' if LAYER_NUM == LAYER_NUM_DEFAULT_VALUE else LAYER_NUM}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    main()
