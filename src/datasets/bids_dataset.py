import pyxdf
import numpy as np
import pandas as pd
import ujson

import mne

import os

from .mne_import_xdf import read_raw_xdf


def read_xdf_eeg_data(config, subject, task):
    # Parameters
    subject_file = "sub-OFS_" + subject
    task_file = "ses-" + task
    xdf_file = "".join(["sub-OFS_", subject, "_ses-", task, "_task-T1_run-001.xdf"])

    # Read paths
    read_path = config["raw_xdf_path"] + subject_file + "/" + task_file + "/" + xdf_file
    raw_eeg, time_info = read_raw_xdf(read_path)
    return raw_eeg, time_info


def read_xdf_eye_data(config, subject, task):
    # Parameters
    subject_file = "sub-OFS_" + subject
    task_file = "ses-" + task
    xdf_file = "".join(["sub-OFS_", subject, "_ses-", task, "_task-T1_run-001.xdf"])

    # Read paths
    read_path = config["raw_xdf_path"] + subject_file + "/" + task_file + "/" + xdf_file
    raw_eye, time_info = read_raw_xdf(read_path, stream_id="Tobii_Eye_Tracker")
    return raw_eye, time_info


def read_individual_diff(config, subject):
    individual_diff_data = {}

    for task in config["individual_diff"]:
        # Save the file
        subject_file = "sub-OFS_" + subject
        read_path = "".join(
            [
                config["raw_xdf_path"],
                subject_file,
                "/",
                task,
                "/",
                task,
                "_OFS_",
                subject,
                ".csv",
            ]
        )
        if task == "MOT":
            mot_df = pd.read_csv(read_path)
            individual_diff_data["mot"] = np.mean(mot_df["N_Reponse"].values) / 4
        else:
            vs_df = pd.read_csv(read_path)
            individual_diff_data["vs"] = np.sum(vs_df["Accuracy"].values) / 30
    return individual_diff_data


def read_xdf_game_data(config, subject, task):
    # Parameters
    subject_file = "sub-OFS_" + subject
    task_file = "ses-" + task
    xdf_file = "".join(["sub-OFS_", subject, "_ses-", task, "_task-T1_run-001.xdf"])

    # Read paths
    read_path = config["raw_xdf_path"] + subject_file + "/" + task_file + "/" + xdf_file

    streams, fileheader = pyxdf.load_xdf(read_path)
    raw_game, time_info = None, None
    for stream in streams:
        if stream["info"]["name"][0] == "parameter_server_states":
            raw_game = [ujson.loads(data[0]) for data in stream["time_series"]]
            time_info = stream
            break
    return raw_game, time_info


def write_edf_file(config, subject, task, data_type):
    if data_type == "eye":
        data, time_info = read_xdf_eye_data(config, subject, task)
    elif data_type == "eeg":
        data, time_info = read_xdf_eeg_data(config, subject, task)
    elif data_type == "game":
        data, time_info = read_xdf_game_data(config, subject, task)

    # Save info
    folder_path = os.path.join(
        config["processed_data_path"],
        f"sub-{subject}",
        f"ses-{config['session']}",
        data_type,
    )
    file_name = f"sub-{subject}_task-{config['tasks'][task]}_run-{config['run']}_{data_type}.edf"
    save_path = os.path.join(folder_path, file_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    mne.export.export_raw(save_path, data, overwrite=config["data_overwrite"])


def convert_to_bids_dataset(config):
    for subject in config["subjects"]:
        for task in config["tasks"].keys():
            for data_type in config["recorded_data_types"]:
                write_edf_file(config, subject, task, data_type)
    return None
