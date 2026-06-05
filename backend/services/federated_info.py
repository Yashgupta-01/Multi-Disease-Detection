import os
import hashlib
from datetime import datetime

# Dummy or static data parsed from what we know about the notebooks
# In a real dynamic scenario, the FL pipeline would write this to a DB or JSON file
FEDERATED_MODELS = {
    "tb": {
        "status": "active",
        "rounds_completed": 10,
        "clients_participated": 3,
        "latest_accuracy": 92.5,
        "last_updated": "2026-04-20T10:00:00Z"
    },
    "skin": {
        "status": "active",
        "rounds_completed": 5,
        "clients_participated": 5,
        "latest_accuracy": 88.4,
        "last_updated": "2026-04-22T14:30:00Z"
    }
}

def get_federated_info():
    """Returns metadata about the federated learning process."""
    return {
        "description": "Federated Learning aggregates models from multiple edge devices without sharing raw patient data.",
        "active_models": FEDERATED_MODELS
    }

def get_federated_status():
    """Returns a hash of the current model weights to verify updates."""
    status_report = {}
    
    # Just generating a fake hash based on file modification time for demonstration.
    # In a real app, you'd hash the actual .pt file.
    model_paths = {
        "tb": "models/tb_cnn.pt",
        "skin": "models/skin_cancer.pt"
    }
    
    for disease, path in model_paths.items():
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            file_hash = hashlib.md5(f"{disease}_{mtime}".encode()).hexdigest()
            status_report[disease] = {
                "global_model_hash": file_hash,
                "timestamp": datetime.fromtimestamp(mtime).isoformat()
            }
        else:
            status_report[disease] = {"error": "Model file not found"}
            
    return status_report
