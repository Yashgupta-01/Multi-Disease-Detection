"""
TB Prediction Service
─────────────────────
Model  : TB_CNN (custom 3-layer CNN, trained in PyTorch)
Input  : Any PIL RGB image
Output : dict with disease, prediction, confidence

Fixes applied (Phase 1):
  - Removed broken `from ML.src.tb_model import TB_CNN` import.
    The import assumed a specific sys.path that does not exist at runtime.
    Class is now defined directly here — this is correct practice for inference services.
  - Normalization kept as [0.5, 0.5, 0.5] to match training exactly.
    (The old main.py had ImageNet stats [0.485, 0.456, 0.406] — that mismatch
    caused wrong predictions even with a correct model.)
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms


class TB_CNN(nn.Module):
    """
    Exactly matches the architecture defined in TB_model.ipynb.
    Do NOT change layer sizes — they must match the saved weights.
    """
    def __init__(self):
        super(TB_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(4, 4)
        self.fc1   = nn.Linear(8 * 56 * 56, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2   = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 8 * 56 * 56)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# ── Load model once at startup (not on every request) ─────────────────────────
device = torch.device("cpu")

model = TB_CNN()
model.load_state_dict(torch.load("models/tb_cnn.pt", map_location=device))
model.eval()

# ── Transform must EXACTLY match what was used during training ─────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

CLASSES = ["Normal", "Tuberculosis"]


def predict_tb(image):
    """
    Args:
        image: PIL.Image in RGB mode
    Returns:
        dict: { disease, prediction, confidence }
    """
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs   = torch.softmax(outputs, dim=1)
        pred    = torch.argmax(probs).item()

    return {
        "disease":    "Tuberculosis",
        "prediction": CLASSES[pred],
        "confidence": round(float(probs[0][pred]) * 100, 2)
    }