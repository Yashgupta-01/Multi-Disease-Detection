import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# ✅ FIXED: This must exactly match the architecture used during training
# The old file had a completely different "BrainCNN" class — that's why predictions were wrong
class BrainMRI_CNN(nn.Module):
    def __init__(self):
        super(BrainMRI_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128 * 26 * 26, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 26 * 26)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device("cpu")

model = BrainMRI_CNN()
model.load_state_dict(torch.load("models/brain_mri_cnn_v1.pt", map_location=device))
model.eval()

# ✅ Added normalization to match training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def predict_brain(image):
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs).item()

    # ✅ Class order matches ImageFolder alphabetical order: 'no_tumor' = 0, 'tumor' = 1
    # But your dataset folders were 'yes' and 'no' — alphabetical: 'no'=0, 'yes'=1
    classes = ['No Tumor', 'Tumor']

    return {
        "disease": "Brain Tumor",
        "prediction": classes[pred],
        "confidence": round(float(probs[0][pred]) * 100, 2)
    }