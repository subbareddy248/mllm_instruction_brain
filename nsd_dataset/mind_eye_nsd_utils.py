from os import path

from nsd_dataset.nsd_gnet8x.src.file_utility import load_mask_from_nii
from nsd_dataset.nsd_gnet8x.src.load_nsd import ordering_split

import scipy.io as sio
import numpy as np
import h5py
from tqdm.auto import tqdm


def load_exp_design_file(base_directory: str):
    exp_design_filepath = path.join(base_directory, "nsd_expdesign.mat")
    return sio.loadmat(exp_design_filepath)


def get_trial_image_orders(base_directory: str, trial_order=None):
    exp_design = load_exp_design_file(base_directory)

    if trial_order is None:
        trial_order = exp_design["masterordering"].flatten() - 1
    else:
        trial_order = trial_order - 1

    subject_idx = exp_design["subjectim"]

    return subject_idx[:, trial_order] - 1


def get_subject_image_ids(base_directory: str, subject: int):
    exp_design = load_exp_design_file(base_directory)
    return exp_design["subjectim"][subject - 1]


def get_subject_images(base_directory: str, subject: int):
    images = load_image_dataset(base_directory)
    subject_image_ids = get_subject_image_ids(base_directory, subject)

    return subject_image_ids, images[subject_image_ids - 1]


def combine_fmri_session_data(
    base_directory: str,
    subject: int,
    sessions: list[int] = range(1, 41),
):
    maskdata = load_mask_from_nii(path.join(base_directory, "nsddata_voxels", f"subj{subject:02}", "nsdgeneral.nii.gz"))
    voxels = np.where(maskdata == 1)

    combined_session_data = None

    for session in tqdm(sessions):
        maindata = load_mask_from_nii(
            path.join(base_directory, "nsddata_sessions", f"subj{subject:02}", f"betas_session{session:02}.nii.gz")
        ).transpose(3, 0, 1, 2)

        current_session_data = maindata[:, voxels[0], voxels[1], voxels[2]]
        if combined_session_data is None:
            combined_session_data = current_session_data
        else:
            combined_session_data = np.concatenate((combined_session_data, current_session_data), axis=0)

    return combined_session_data


def get_split_data(
    base_directory: str,
    subject: int,
    sessions: list[int] = range(1, 41),
    average_out_fmri: bool = False,
):
    exp_design = load_exp_design_file(base_directory)
    ordering = exp_design["masterordering"].flatten() - 1

    with open(path.join(base_directory, "nsddata_sessions", f"subj{subject:02}", "combined_sessions.npy"), "rb") as f:
        combined_session_data = np.load(f)

    if average_out_fmri:
        ordering_done = [False] * (ordering.max() + 1)
        new_ordering = []
        order_to_fmri = {}
        for indx, ord in enumerate(ordering):
            if not ordering_done[ord]:
                new_ordering.append(ord)
                ordering_done[ord] = True
                order_to_fmri[ord] = [np.copy(combined_session_data[indx])]
            else:
                order_to_fmri[ord].append(np.copy(combined_session_data[indx]))

        combined_session_data = []
        for ord in new_ordering:
            combined_session_data.append(np.array(order_to_fmri[ord]).mean(axis=0))

        combined_session_data = np.array(combined_session_data)
        ordering = np.array(new_ordering)

    return (
        ordering + 1,
        combined_session_data,
        ordering_split(
            np.copy(combined_session_data),
            ordering,
            combine_trial=False,
        ),
    )


def load_image_dataset(base_directory: str):
    dataset_filepath = path.join(base_directory, "nsddata_stimuli", "nsd_stimuli_227.hdf5")
    return np.array(h5py.File(dataset_filepath, "r")["imgBrick"])
