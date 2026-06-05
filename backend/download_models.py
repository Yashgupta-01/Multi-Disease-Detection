import os
import gdown

# Map the exact required filenames to their Google Drive file IDs
MODELS_TO_DOWNLOAD = {
    "brain_mri_cnn_v1.pt": "1IrszLYdJFxktvZJOdTwMH9wkBRef7-Gz",
    "skin_cancer.pt": "10ccF3Tj5Z_KW34iSsKs1Wy2cRfpTKGwv",
    "oct_cnn.pt": "1yhrB6GP0n-WknuREhtG0VfiTDTFe8v7J",
    "tb_cnn.pt": "1tPRUf5-NmPhy3nNuvaGLole-3lRmsk8w"
}

MODEL_DIR = "models"

def download_models():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    for filename, file_id in MODELS_TO_DOWNLOAD.items():
        output_path = os.path.join(MODEL_DIR, filename)
        
        # Skip download if the file already exists (saves time on reboot if persistent disk is used)
        if os.path.exists(output_path):
            print(f"✅ {filename} already exists. Skipping download.")
            continue
            
        print(f"📥 Downloading {filename} from Google Drive...")
        
        try:
            # We explicitly set the output filename so it ignores the Google Drive name
            downloaded_file = gdown.download(id=file_id, output=output_path, quiet=False)
            if downloaded_file:
                print(f"🎉 Successfully downloaded: {downloaded_file}")
            else:
                print(f"❌ Failed to download {filename}.")
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")

if __name__ == "__main__":
    download_models()
