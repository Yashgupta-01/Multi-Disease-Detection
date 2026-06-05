"""
Skin Cancer Prediction Service
────────────────────────────────
Model  : Custom CNN (4 conv-blocks + BN + classifier), trained in PyTorch
         Binary: Benign (class 0) / Malignant (class 1) — softmax output
Input  : Any PIL RGB image
Output : dict with disease, prediction, confidence

Architecture reverse-engineered from skin_cancer.pt state_dict:
  features:
    0  Conv2d(3 → 32, k=3, padding=1)        [features.0]
    1  BatchNorm2d(32)                         [features.1]
    2  ReLU
    3  MaxPool2d(2, 2)
    4  Conv2d(32 → 64, k=3, padding=1)        [features.4]
    5  BatchNorm2d(64)                         [features.5]
    6  ReLU
    7  MaxPool2d(2, 2)
    8  Conv2d(64 → 128, k=3, padding=1)       [features.8]
    9  BatchNorm2d(128)                        [features.9]
    10 ReLU
    11 MaxPool2d(2, 2)
    12 Conv2d(128 → 256, k=3, padding=1)      [features.12]
    13 BatchNorm2d(256)                        [features.13]
    14 ReLU
    15 MaxPool2d(2, 2)
  classifier (AdaptiveAvgPool → Flatten → FC layers):
    0  AdaptiveAvgPool2d((14, 14))             → 256*14*14 = 50176
    1  Flatten
    1  Linear(50176 → 512)                    [classifier.1]
    2  ReLU
    3  Dropout
    4  Linear(512 → 128)                      [classifier.4]
    5  ReLU
    6  Dropout
    7  Linear(128 → 2)                        [classifier.7]

Notes:
  - Normalization uses ImageNet stats to match training pipeline.
  - Model produces 2 logits → softmax for confidence.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms


# ── Architecture (must exactly mirror training notebook) ──────────────────────

class SkinCancerCNN(nn.Module):
    """
    4-block CNN with BatchNorm, followed by a 3-layer classifier.
    MUST match the architecture used when skin_cancer.pt was trained.
    Do NOT change layer sizes — they are fixed by the saved weights.
    """
    def __init__(self):
        super(SkinCancerCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1 — indices 0,1,2,3
            nn.Conv2d(3, 32, kernel_size=3, padding=1),    # features.0
            nn.BatchNorm2d(32),                             # features.1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2 — indices 4,5,6,7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # features.4
            nn.BatchNorm2d(64),                             # features.5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3 — indices 8,9,10,11
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # features.8
            nn.BatchNorm2d(128),                            # features.9
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4 — indices 12,13,14,15
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # features.12
            nn.BatchNorm2d(256),                            # features.13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                                   # classifier.0 (not in state_dict)
            nn.Linear(256 * 14 * 14, 512),                 # classifier.1  → 50176 → 512
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),                           # classifier.4
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2),                             # classifier.7
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── Load model once at startup ─────────────────────────────────────────────────
device = torch.device("cpu")

model = SkinCancerCNN()
model.load_state_dict(torch.load("models/skin_cancer.pt", map_location=device))
model.eval()

# ── Transform — ImageNet normalization to match training ───────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

CLASSES = ["Benign", "Malignant"]


def predict_skin(image):
    """
    Args:
        image: PIL.Image in RGB mode (passed by main.py)
    Returns:
        dict: { disease, prediction, confidence }
    """
    img = transform(image).unsqueeze(0).to(device)   # Shape: [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(img)                          # Logits: [1, 2]
        probs   = torch.softmax(outputs, dim=1)       # Probabilities
        pred    = torch.argmax(probs).item()          # 0 = Benign, 1 = Malignant

    return {
        "disease":    "Skin Cancer",
        "prediction": CLASSES[pred],
        "confidence": round(float(probs[0][pred]) * 100, 2)
    }