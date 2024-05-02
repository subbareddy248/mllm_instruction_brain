# mind_eye

## Downloading the dataset

* The dataset can be found at [this link](https://drive.google.com/drive/folders/1lNlV7EKRidd4H-xWTs0kaIR-bHrAyxkq?usp=sharing)
* Download the dataset and note down the path to the directory where all the nsd_* dataset directories are present. We will need to put it in the script later on.

## Conda Environment

* Use the `environment.yml` file to create the conda environment with all the dependencies. You might have to change the CUDA version to the one on your local device.


## Running the scripts

* Go to the `./brain_activity_prediction/extract_image_embeddings/` directory
* Open and edit the `run_experiments.sh` file:
    - Change the value of the `BASE_DIRECTORY` variable to the path of the dataset folder. This is also the directory where the hidden states will be put.
    - Change the value of the `GPU_ID` variable to the CUDA ID number of the GPU. For example `1`.
* Then just run the script with `./run_experiments.sh`.
