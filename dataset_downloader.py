import os.path
import zipfile

import gdown


def is_exist(path):
    return os.path.exists(path)


def generate_direct_url(file_id):
    return f"https://drive.google.com/uc?id={file_id}"


print("Downloading Dataset Please Wait...")
dataset_drive_id = "117-b_rYGnckqu2XnfTZXWDGh8Tkf4wak"
output_path = "face_mask_detection_dataset.zip"
dataset_download_url = generate_direct_url(dataset_drive_id)

gdown.download(dataset_download_url, output_path, quiet=False)

if not is_exist("dataset/"):
    os.mkdir("dataset/")

if is_exist(output_path):
    print("Extracting Dataset ZIP File...")
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall("dataset/")
    os.remove(output_path)

print("Training Dataset Downloaded And Extracted To 'dataset/' Directory.")
