"""
OCT Eye Disease Prediction Service
────────────────────────────────────
Model  : OCTNet (custom 2-conv CNN, trained in PyTorch)
         This is the best-performing model in the project (97% test accuracy).
Input  : Any PIL RGB image
Output : dict with disease, prediction, confidence

Fixes applied (Phase 1):
  - CRITICAL FIX: Normalize([0.5, 0.5, 0.5]) was completely missing.
    The model was trained on normalized data but receiving raw [0,1] pixel
    values during inference. Adding this one transform is the entire fix.
  - Also fixed a stray typo in the training notebook (float('inf')wdfw)
    — not relevant here but noted for reference.

No retraining needed — OCTNet is already the best model at 97%.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class OCTNet(nn.Module):
    """
    Matches Human_eye_disease_model.ipynb exactly.
    Input: 224x224 RGB
    After conv1 + pool(2,2): 16 x 112 x 112
    After conv2 + pool(2,2): 32 x 56 x 56
    Flatten: 32 * 56 * 56 = 100352
    """
    def __init__(self):
        super(OCTNet, self).__init__()
        self.conv1   = nn.Conv2d(3,  16, 3, padding=1)
        self.conv2   = nn.Conv2d(16, 32, 3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1     = nn.Linear(32 * 56 * 56, 256)
        self.fc2     = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ── Load model once at startup ─────────────────────────────────────────────────
device = torch.device("cpu")

model = OCTNet()
model.load_state_dict(torch.load("models/oct_cnn.pt", map_location=device))
model.eval()

# ── Transform — Normalize was THE missing piece ────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 4-class classification — order matches training ImageFolder (alphabetical):
# CNV=0, DME=1, DRUSEN=2, NORMAL=3
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def predict_oct(image):
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
        "disease":    "Eye Disease (OCT)",
        "prediction": CLASSES[pred],
        "confidence": round(float(probs[0][pred]) * 100, 2)
    }