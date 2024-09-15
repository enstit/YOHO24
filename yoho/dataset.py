#!/usr/bin/env python

import os
import logging
import argparse

from yoho.utils import (
    download_file,
    uncompress_file,
    get_files,
    write_data_to_csv,
)


def join_paths(*paths):
    return os.path.abspath(os.path.join(*paths))


SCRIPT_DIRPATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = join_paths(SCRIPT_DIRPATH, "../data/")

logging.basicConfig(level=logging.INFO)

"""
def parse_jams_file(jams_file):
    \"""
    Parse a JAMS file and extract the annotations.

    Parameters:
    - jams_file (str): The path to the JAMS file.

    Returns:
    - events (list): A list of tuples containing the event type, start time, and end time.
    \"""
    jam = jams.load(jams_file)
    events = []
    for annotation in jam.annotations:

        for obs in annotation.data:

            start_time = obs.value["event_time"]
            end_time = obs.value["event_time"] + obs.value["event_duration"]
            label = obs.value["label"]
            events.append((label, start_time, end_time))

        return events
"""


def download_urbansed():
    DATASET_URL = "https://zenodo.org/api/records/1324404/files-archive"
    urbansed_raw_path = join_paths(RAW_PATH, "UrbanSED/")
    urbansed_zip_file = join_paths(urbansed_raw_path, "UrbanSED.zip")

    if not os.path.exists(urbansed_raw_path):
        os.makedirs(urbansed_raw_path)

        if not os.path.exists(urbansed_zip_file):
            download_file(DATASET_URL, urbansed_zip_file)

        uncompress_file(urbansed_zip_file, urbansed_raw_path)
        uncompress_file(
            join_paths(urbansed_raw_path, "URBAN-SED_v2.0.0.tar.gz"),
            urbansed_raw_path,
        )
        # Move all the files to the parent folder
        for item in os.listdir(
            join_paths(urbansed_raw_path, "URBAN-SED_v2.0.0")
        ):
            item_path = join_paths(urbansed_raw_path, "URBAN-SED_v2.0.0", item)
            os.rename(item_path, join_paths(urbansed_raw_path, item))
        # Remove the empty folder
        os.rmdir(join_paths(urbansed_raw_path, "URBAN-SED_v2.0.0"))


def process_urbansed():
    urbansed_processed_path = join_paths(PROCESSED_PATH, "UrbanSED/")

    if not os.path.exists(urbansed_processed_path):
        os.makedirs(urbansed_processed_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tut",
        action="store_true",  # default=False
        help="Download the TUT Sound Events 2017 dataset",
    )

    parser.add_argument(
        "--urbansed",
        action="store_true",  # default=False
        help="Download the UrbanSED dataset",
    )

    args = parser.parse_args()

    if not args.tut and not args.urbansed:
        print("Please specify a dataset to download.")
        exit()

    RAW_PATH = join_paths(DATA_PATH, "raw/")
    # Create directories in the data folder if they don't exist
    if not os.path.exists(RAW_PATH):
        logging.info(f"Creating directory: {RAW_PATH}")
        os.makedirs(RAW_PATH)

    PROCESSED_PATH = join_paths(DATA_PATH, "processed/")
    if not os.path.exists(PROCESSED_PATH):
        logging.info(f"Creating directory: {PROCESSED_PATH}")
        os.makedirs(PROCESSED_PATH)

    if args.tut:
        pass

    if args.urbansed:
        download_urbansed()
        process_urbansed()

    """


    ANNOTATIONS_TRAIN_PATH = os.path.join(
        urban_sed_subfolder, "annotations/train/"
    )

    ANNOTATIONS_VALIDATE_PATH = os.path.join(
        urban_sed_subfolder, "annotations/validate/"
    )

    ANNOTATIONS_TEST_PATH = os.path.join(
        urban_sed_subfolder, "annotations/test/"
    )

    AUDIO_TRAIN_PATH = os.path.join(urban_sed_subfolder, "audio/train/")

    AUDIO_TEST_PATH = os.path.join(urban_sed_subfolder, "audio/test/")

    AUDIO_VAL_PATH = os.path.join(urban_sed_subfolder, "audio/validate/")

    train_files = get_files(ANNOTATIONS_TRAIN_PATH, extensions=".jams")
    validate_files = get_files(ANNOTATIONS_VALIDATE_PATH, extensions=".jams")
    test_files = get_files(ANNOTATIONS_TEST_PATH, extensions=".jams")

    # Process the training data
    urbansed_data = {}
    for f in train_files:
        f_path = ANNOTATIONS_TRAIN_PATH + f
        f = f.split(".")[0] + ".wav"
        urbansed_data[AUDIO_TRAIN_PATH + f] = parse_jams_file(f_path)

    write_data_to_csv(
        urbansed_data,
        os.path.join(
            SCRIPT_DIRPATH,
            "../data/processed/URBAN-SED/URBAN-SED_train.csv",
        ),
    )

    # Process the validation data
    urbansed_data = {}
    for f in validate_files:
        f_path = ANNOTATIONS_VALIDATE_PATH + f
        f = f.split(".")[0] + ".wav"
        urbansed_data[AUDIO_VAL_PATH + f] = parse_jams_file(f_path)

    write_data_to_csv(
        urbansed_data,
        os.path.join(
            SCRIPT_DIRPATH,
            "../data/processed/URBAN-SED/URBAN-SED_validate.csv",
        ),
    )

    # Process the test data
    urbansed_data = {}
    for f in test_files:
        f_path = ANNOTATIONS_TEST_PATH + f
        f = f.split(".")[0] + ".wav"
        urbansed_data[AUDIO_TEST_PATH + f] = parse_jams_file(f_path)

    write_data_to_csv(
        urbansed_data,
        os.path.join(
            SCRIPT_DIRPATH,
            "../data/processed/URBAN-SED/URBAN-SED_test.csv",
        ),
    )
    """

    """# TUT Sound Events 2017 Dataset
    tut_urls = [
        (
            "TUT-sound-events-2017-development",
            "https://zenodo.org/api/records/814831/files-archive",
        ),
        (
            "TUT-sound-events-2017-evaluation",
            "https://zenodo.org/api/records/1040179/files-archive",
        ),
    ]
    tut_raw_folder = os.path.join(SCRIPT_DIRPATH, "../data/raw/TUT")

    for tut_name, tut_url in tut_urls:

        if not os.path.exists(tut_raw_folder):
            os.makedirs(tut_raw_folder)

        archive_filepath = os.path.join(tut_raw_folder, f"{tut_name}.zip")
        tut_extract_to_subfolder = os.path.join(tut_raw_folder, tut_name)
        download_file(tut_url, archive_filepath)
        uncompress_file(archive_filepath, tut_extract_to_subfolder)

        for item in os.listdir(tut_extract_to_subfolder):
            if item.endswith(".zip"):
                zipped_file = os.path.join(tut_extract_to_subfolder, item)
                unzipped_file = zipped_file.rsplit(".", 1)[
                    0
                ]  # Remove .zip extension
                uncompress_file(zipped_file, unzipped_file)

    if not os.path.exists(os.path.join(SCRIPT_DIRPATH, "../data/processed")):
        os.makedirs(os.path.join(SCRIPT_DIRPATH, "../data/processed"))
    if not os.path.exists(
        os.path.join(SCRIPT_DIRPATH, "../data/processed/TUT")
    ):
        os.makedirs(os.path.join(SCRIPT_DIRPATH, "../data/processed/TUT"))

    AUDIO_1_PATH = os.path.join(
        SCRIPT_DIRPATH,
        "../data/raw/TUT/TUT-sound-events-2017-development/TUT-sound-events-2017-development.audio.1/TUT-sound-events-2017-development/audio/street/",
    )
    AUDIO_2_PATH = os.path.join(
        SCRIPT_DIRPATH,
        "../data/raw/TUT/TUT-sound-events-2017-development/TUT-sound-events-2017-development.audio.2/TUT-sound-events-2017-development/audio/street/",
    )
    DATA_PATH = os.path.join(SCRIPT_DIRPATH, "../data/processed/TUT/")

    DEVELOPMENT_ANNOTATIONS_PATH = os.path.join(
        SCRIPT_DIRPATH,
        "../data/raw/TUT/TUT-sound-events-2017-development/TUT-sound-events-2017-development.meta/TUT-sound-events-2017-development/meta/street/",
    )

    files = get_files(DEVELOPMENT_ANNOTATIONS_PATH, extensions=".ann")

    tut_data_train = {}
    for f in files:
        with open(DEVELOPMENT_ANNOTATIONS_PATH + f, "r"):

            f_name = f.split(".")[0] + ".wav"

            if f_name in ["a128.wav", "a131.wav", "b007.wav", "b093.wav"]:
                f_path = os.path.abspath(os.path.join(AUDIO_2_PATH + f_name))
            else:
                f_path = os.path.abspath(os.path.join(AUDIO_1_PATH + f_name))

            tut_data_train[f_path] = []

            print(f"Processing {f_path}...")

            with open(DEVELOPMENT_ANNOTATIONS_PATH + f, "r") as file:

                reader = csv.reader(file)

                for row in reader:
                    if row:
                        # split in \t and get the start and end time
                        row = row[0].split("\t")
                        start = float(row[2])
                        end = float(row[3])
                        label = row[4]
                        tut_data_train[f_path].append((label, start, end))

    write_data_to_csv(
        tut_data_train,
        os.path.join(
            SCRIPT_DIRPATH,
            "../data/processed/TUT/TUT-sound-events-2017-development.csv",
        ),
    )

    import soundfile as sf

    DATA_PATH = os.path.join(DATA_PATH, "TUT-sound-events-2017-development")

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    # Process the data
    for f in tut_data_train.keys():
        audio = AudioFile(filepath=f, labels=[])
        audioclips = audio.subdivide(win_len=2.56, hop_len=1.96)

        for i, audioclip in enumerate(audioclips):
            sf.write(
                os.path.join(
                    DATA_PATH, f"{os.path.basename(f).split('.')[0]}_{i}.wav"
                ),
                data=audioclip.waveform,
                samplerate=audioclip.sr,
            )

    AUDIO_1_PATH = os.path.join(
        SCRIPT_DIRPATH,
        "../data/raw/TUT/TUT-sound-events-2017-evaluation/TUT-sound-events-2017-evaluation.audio/TUT-sound-events-2017-evaluation/audio/street/",
    )
    DATA_PATH = os.path.join(SCRIPT_DIRPATH, "../data/processed/TUT/")

    EVALUATION_ANNOTATIONS_PATH = os.path.join(
        SCRIPT_DIRPATH,
        "../data/raw/TUT/TUT-sound-events-2017-evaluation/TUT-sound-events-2017-evaluation.meta/TUT-sound-events-2017-evaluation/meta/street/",
    )

    files = get_files(EVALUATION_ANNOTATIONS_PATH, extensions=".ann")

    tut_data_evaluation = {}
    for f in files:
        with open(EVALUATION_ANNOTATIONS_PATH + f, "r"):

            f_name = f.split(".")[0] + ".wav"
            f_path = AUDIO_1_PATH + f_name

            tut_data_evaluation[f_path] = []

            print(f"Processing {f_path}...")

            with open(EVALUATION_ANNOTATIONS_PATH + f, "r") as file:

                reader = csv.reader(file)

                for row in reader:
                    if row:
                        # split in \t and get the start and end time
                        row = row[0].split("\t")
                        start = float(row[0])
                        end = float(row[1])
                        label = row[2]
                        tut_data_evaluation[f_path].append((label, start, end))

    write_data_to_csv(
        tut_data_evaluation,
        os.path.join(
            SCRIPT_DIRPATH,
            "../data/processed/TUT/TUT-sound-events-2017-evaluation.csv",
        ),
    )"""
