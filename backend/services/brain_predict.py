"""
Brain MRI Prediction Service
─────────────────────────────
Model  : BrainMRI_CNN (3-conv-layer CNN, trained in PyTorch)
Input  : Any PIL RGB image
Output : dict with disease, prediction, confidence

Fixes applied (Phase 1):
  - CRITICAL BUG FIXED: The old file defined a completely different class
    called `BrainCNN` with 2 conv layers and wrong filter sizes.
    The saved weights were trained on `BrainMRI_CNN` (3 conv layers).
    PyTorch was silently loading weights into the wrong architecture,
    which is why brain predictions were always wrong.
  - Added Normalize([0.5, 0.5, 0.5]) to match training transform.

Known issue (Phase 3 — needs retraining):
  - Model overfits: train acc ~95% vs val acc ~72%.
  - Fix: add Dropout(0.5) after fc1 in the notebook and retrain.
    After retraining, replace models/brain_mri_cnn_v1.pt.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class BrainMRI_CNN(nn.Module):
    """
    MUST match brain_model.ipynb exactly.
    Input image: 224x224 RGB
    After conv1 + pool(2,2): 32 x 111 x 111
    After conv2 + pool(2,2): 64 x 54 x 54  — wait, let's trace carefully:
      224 → conv(3,32,k=3) → 222 → pool(2,2) → 111
      111 → conv(32,64,k=3) → 109 → pool(2,2) → 54
      54  → conv(64,128,k=3) → 52 → pool(2,2) → 26
    Flatten: 128 * 26 * 26 = 86528
    """
    def __init__(self):
        super(BrainMRI_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,   32,  kernel_size=3)
        self.conv2 = nn.Conv2d(32,  64,  kernel_size=3)
        self.conv3 = nn.Conv2d(64,  128, kernel_size=3)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 26 * 26, 128)
        self.fc2   = nn.Linear(128, 2)
        # NOTE: Dropout will be added here after Phase 3 retraining.
        # self.dropout = nn.Dropout(0.5)  ← uncomment after retrain

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 26 * 26)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)  ← uncomment after Phase 3 retrain
        return self.fc2(x)


# ── Load model once at startup ─────────────────────────────────────────────────
device = torch.device("cpu")
model = None

def get_model():
    global model
    if model is None:
        model = BrainMRI_CNN()
        model.load_state_dict(torch.load("models/brain_mri_cnn_v1.pt", map_location=device))
        model.eval()
    return model

# ── Transform matches training ─────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ImageFolder loads classes alphabetically.
# Dataset folders: 'no_tumor' and 'tumor' → alphabetical: no_tumor=0, tumor=1
CLASSES = ["No Tumor", "Tumor"]


def predict_brain(image):
    """
    Args:
        image: PIL.Image in RGB mode
    Returns:
        dict: { disease, prediction, confidence }
    """
    m = get_model()
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = m(img)
        probs   = torch.softmax(outputs, dim=1)
        pred    = torch.argmax(probs).item()

    return {
        "disease":    "Brain Tumor",
        "prediction": CLASSES[pred],
        "confidence": round(float(probs[0][pred]) * 100, 2)
    }