import os
import csv
import requests
import zipfile
import tarfile
from tqdm import tqdm
import librosa


def get_files(data_path: str, extensions: str) -> list:
    """
    Get a list of files in the specified data path with the given extensions.

    Parameters:
    - data_path (str): The path to the directory containing the files.
    - extensions (str or tuple): The file extensions to filter by.

    Returns:
    - files (list): A list of file names that match the specified extensions.
    """
    return [f for f in os.listdir(data_path) if f.endswith(extensions)]


def write_data_to_csv(data: dict, output_path: str) -> None:
    """
    Takes a dictionary and writes it to a CSV file.

    Parameters:
    - data (dict): A dictionary containing the data to be written to the CSV file.
    - output_path (str): The path to the output CSV file.
    """
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filepath', 'events'])

        for key, value in data.items():
            writer.writerow([key, value])


def download_file(url: str, save_path: str) -> None:
    """
    Download a file from a URL and save it locally.

    Parameters:
    - url (str): URL of the file to download.
    - save_path (str): Local path to save the downloaded file.
    """
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as file:
        for chunk in tqdm(response.iter_content(chunk_size=128)):
            file.write(chunk)
    print(f"Downloaded {url} to {save_path}")


def uncompress_file(compressed_path: str, extract_to: str) -> None:
    """
    Uncompress a file to a specified directory.
    Supported file types: .zip, .tar, .tar.gz, .tar.bz2

    Parameters:
    - compressed_path (str): Path of the compressed file to uncompress.
    - extract_to (str): Directory to save the uncompress file.
    """
    if compressed_path.endswith('.zip'):
        with zipfile.ZipFile(compressed_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif compressed_path.endswith('.tar') or compressed_path.endswith('.tar.gz') or compressed_path.endswith('.tar.bz2'):
        with tarfile.open(compressed_path, 'r') as tar_ref:
            tar_ref.extractall(extract_to)
    print(f"Extracted {compressed_path} to {extract_to}")
    os.remove(compressed_path)


def get_audio_duration(filepath: str) -> float:
    """
    Get the duration of an audio file in seconds.

    Parameters:
    - filepath (str): Path to the audio file.

    Returns:
    - duration (float): Duration of the audio file in seconds.
    """
    return librosa.get_duration(filename=filepath)
