import os
import gdown

# The 4 file IDs provided
FILE_IDS = [
    "1tPRUf5-NmPhy3nNuvaGLole-3lRmsk8w",
    "1yhrB6GP0n-WknuREhtG0VfiTDTFe8v7J",
    "1W491aguZsTK8cm129iFIytPyYWoAVqn1",
    "10ccF3Tj5Z_KW34iSsKs1Wy2cRfpTKGwv"
]

MODEL_DIR = "models"

def download_models():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    for file_id in FILE_IDS:
        url = f"https://drive.google.com/uc?id={file_id}"
        # We don't specify output name so it takes the original filename from Drive
        output = os.path.join(MODEL_DIR, file_id) # Temporary name, we will let gdown resolve it
        print(f"Downloading {file_id}...")
        
        # gdown will automatically extract the filename if output is a directory
        # but to be safe, we let gdown figure out the filename and put it in MODEL_DIR
        try:
            downloaded_file = gdown.download(id=file_id, output=f"{MODEL_DIR}/", quiet=False)
            print(f"Successfully downloaded: {downloaded_file}")
        except Exception as e:
            print(f"Error downloading {file_id}: {e}")

if __name__ == "__main__":
    download_models()
