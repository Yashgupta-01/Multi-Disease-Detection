import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ✅ TB_CNN defined here directly — no import conflict
class TB_CNN(nn.Module):
    def __init__(self):
        super(TB_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(8 * 56 * 56, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 8 * 56 * 56)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device("cpu")

model = TB_CNN()
model.load_state_dict(torch.load("models/tb_cnn.pt", map_location=device))
model.eval()

# ✅ Normalization matches training exactly
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def predict_tb(image):
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs).item()

    classes = ['Normal', 'Tuberculosis']

    return {
        "disease": "Tuberculosis",
        "prediction": classes[pred],
        "confidence": round(float(probs[0][pred]) * 100, 2)
    }