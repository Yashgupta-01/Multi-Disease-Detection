import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# ✅ OCTNet defined directly — no sys.path hacks needed
class OCTNet(nn.Module):
    def __init__(self):
        super(OCTNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device("cpu")

model = OCTNet()
model.load_state_dict(torch.load("models/oct_cnn.pt", map_location=device))
model.eval()

# ✅ FIXED: Added Normalize — was missing before, causing wrong predictions
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

def predict_oct(image):
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs).item()

    return {
        "disease": "Eye Disease (OCT)",
        "prediction": classes[pred],
        "confidence": round(float(probs[0][pred]) * 100, 2)
    }