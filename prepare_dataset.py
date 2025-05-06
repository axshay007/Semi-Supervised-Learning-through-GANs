import os
import shutil
import urllib.request
import zipfile
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar."""
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(
            url, filename,
            reporthook=lambda count, block_size, total_size: t.update(block_size)
        )

def prepare_dataset():
    """Download and prepare the HAM10000 dataset for training."""
    # Dataset URL
    url = "https://lp-prod-resources.s3.amazonaws.com/278/45149/2021-02-19-19-47-43/MelanomaDetection.zip"
    zip_path = "MelanomaDetection.zip"
    
    # Download dataset if not exists
    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        download_file(url, zip_path)
    
    # Extract dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("temp_data")
    
    # Create directory structure
    os.makedirs("data/labeled/benign", exist_ok=True)
    os.makedirs("data/labeled/malignant", exist_ok=True)
    os.makedirs("data/unlabeled", exist_ok=True)
    os.makedirs("data/test/benign", exist_ok=True)
    os.makedirs("data/test/malignant", exist_ok=True)
    
    # Move training data
    print("Organizing training data...")
    train_dir = "temp_data/MelanomaDetection/MelanomaDetection/Train"
    for label in ["benign", "malignant"]:
        src_dir = os.path.join(train_dir, label)
        dst_dir = f"data/labeled/{label}"
        for img in os.listdir(src_dir):
            shutil.copy2(os.path.join(src_dir, img), os.path.join(dst_dir, img))
    
    # Move test data
    print("Organizing test data...")
    test_dir = "temp_data/MelanomaDetection/MelanomaDetection/Test"
    for label in ["benign", "malignant"]:
        src_dir = os.path.join(test_dir, label)
        dst_dir = f"data/test/{label}"
        for img in os.listdir(src_dir):
            shutil.copy2(os.path.join(src_dir, img), os.path.join(dst_dir, img))
    
    # Move unlabeled data
    print("Organizing unlabeled data...")
    unlabeled_dir = "temp_data/MelanomaDetection/MelanomaDetection/unlabeled"
    for img in os.listdir(unlabeled_dir):
        shutil.copy2(os.path.join(unlabeled_dir, img), os.path.join("data/unlabeled", img))
    
    # Clean up
    print("Cleaning up...")
    shutil.rmtree("temp_data")
    os.remove(zip_path)
    
    print("Dataset preparation completed!")

if __name__ == "__main__":
    prepare_dataset() 