# mind_eye

## Downloading the dataset

* The dataset can be found at [this link](https://drive.google.com/drive/folders/1lNlV7EKRidd4H-xWTs0kaIR-bHrAyxkq?usp=sharing)
* Download the dataset and note down the path to the directory where all the nsd_* dataset directories are present. We will need to put it in the script later on.

## Conda Environment

* Use the `environment.yml` file to create the conda environment with all the dependencies. You might have to change the CUDA version to the one on your local device.


## Running the scripts

### Extraction


#### Using the helper script for extraction

* Go to the `./brain_activity_prediction/extract_image_embeddings/` directory
* All the current prompts will be in the `prompts.py` file there.
* Open and edit the `run_experiments.sh` file:
    - Change the value of the `BASE_DIRECTORY` variable to the path of the dataset folder. This is also the directory where the hidden states will be put.
    - Change the value of the `GPU_ID` variable to the CUDA ID number of the GPU. For example `1`.
    - You can change which prompts to run the experiments for in line 27 (`for prompt_number in {1..9}; do`). Currently it will run for all the 9 prompts.
* Then just run the script with `./run_experiments.sh`.

#### Manually running the extraction scripts

* Go to the `./brain_activity_prediction/extract_image_embeddings/` directory.
* In this directory, there are scripts present for each model:
    1. `idefics_9b_instruct.py`
    2. `instruct_blip_13b.py`
    3. `llava_1.5_13b.py`
    4. `mplug_owl_llama_7b.py`
    5. `openflamingo_9b.py`
    6. `vit_huge_patch14_224_in21k.py`
* All of them take the following flags as input:
 ```
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The number of images which will be passed into the model together
  -d BASE_DIR, --base-dir BASE_DIR
                        The path to the directory where all the inputs and the outputs will be cached and loaded from
  -s {1,2,5,7}, --subject {1,2,5,7}
                        The subject number from the NSD dataset whose images' embeddings are to be extracted
  -t, --test-run, --no-test-run
                        Enable test-run to just output the outputs for the first batch and exit
  -p PROMPT_NUMBER, --prompt-number PROMPT_NUMBER
                        The number of the prompt from prompts.py that will be passed into the model along with the images
  -g GPU_ID, --gpu-id GPU_ID
                        The CUDA GPU id on which to run inference
```
* In all the scripts, `BASE_DIR` is basically path to the directory which contains all the `nsd_` directories from dataset. All the outputs will also be extracted into this folder.

* All the scripts output the hidden layers in different batches inside the `BASE_DIR/image_embeddings` directory. They are batchified so that if there is a failure in between then the program can continue from a saved point automatically.

### Alignment

* Go to the `./brain_activity_prediction/align_image_embeddings/` directory.
* Here, you will find the `align_model.py` script which takes the following flags as input:
```
  -s {1,2,5,7}, --subject {1,2,5,7}
                        The subject number from the NSD dataset whose image embeddings are to be trained
  -d BASE_DIR, --base-dir BASE_DIR
                        The path to the directory where all the models, inputs and outputs will be stored and loaded from
  -m MODEL_ID, --model-id MODEL_ID
                        The model id whose hidden state representations are to be used
  -p PROMPT_NUMBER, --prompt-number PROMPT_NUMBER
                        The prompt number to use for aligning
  --max-log-10-alpha MAX_LOG_10_ALPHA
                        Maximum value of log10 alpha to consider, default is 4 (basically till 10^4)
  --num-alphas NUM_ALPHAS
                        Number of alpha values to samples, default 60
```
* In this, `MODEL_ID` can be:
    1. `google/vit-huge-patch14-224-in21k`
    2. `Salesforce/instructblip-vicuna-13b`
    3. `HuggingFaceM4/idefics-9b-instruct`
    4. `llava-hf/llava-1.5-13b-hf`
    5. `MAGAer13/mplug-owl-llama-7b`
    6. `openflamingo/OpenFlamingo-9B-vitl-mpt7b`
* Around the line 50 in this file, you can change the parameters to fully utilize the GPU
