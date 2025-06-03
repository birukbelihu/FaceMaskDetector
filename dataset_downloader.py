import os.path
import zipfile

import gdown

dataset_drive_id = "10LmdTEkcuRBtA5CtImAoHiTuxRcgvq3U"
output_path = "face_mask_detection_dataset.zip"

if not os.path.exists("dataset/"):
    os.mkdir("dataset/")

dataset_download_url = f"https://drive.google.com/uc?id={dataset_drive_id}"
gdown.download(dataset_download_url, output_path, quiet=False)

with zipfile.ZipFile(output_path, 'r') as zip_ref:
    zip_ref.extractall("dataset/")

print("Dataset downloaded and extracted to 'dataset/' directory.")
os.remove(output_path)
