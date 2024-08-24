#!/usr/bin/env python

import os
from utils import download_file, uncompress_file

SCRIPT_DIRPATH = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":

    # TUT Sound Events 2017 Dataset
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
    tut_zip_path = os.path.join(SCRIPT_DIRPATH, "../data/tut.zip")
    tut_extract_to = os.path.join(SCRIPT_DIRPATH, "../data/tut")

    for tut_name, tut_url in tut_urls:
        tut_extract_to_subfolder = os.path.join(tut_extract_to, tut_name)
        download_file(tut_url, tut_zip_path)
        uncompress_file(tut_zip_path, tut_extract_to_subfolder)

        for item in os.listdir(tut_extract_to_subfolder):
            if item.endswith(".zip"):
                zipped_file = os.path.join(tut_extract_to_subfolder, item)
                unzipped_file = zipped_file.rsplit(".", 1)[0]  # Remove .zip extension
                uncompress_file(zipped_file, unzipped_file)
