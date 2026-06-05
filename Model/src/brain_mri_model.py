import torch.nn as nn
import torch.nn.functional as F

class BrainMRI_CNN(nn.Module):
    def __init__(self):
        super(BrainMRI_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128 * 26 * 26, 128)  # Adjusted for 224x224 input
        self.fc2 = nn.Linear(128, 2)  # 2 classes: Tumor / No Tumor

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # Conv1 + ReLU + Pool
        x = self.pool(F.relu(self.conv2(x)))   # Conv2 + ReLU + Pool
        x = self.pool(F.relu(self.conv3(x)))   # Conv3 + ReLU + Pool
        x = x.view(-1, 128 * 26 * 26)           # Flatten
        x = F.relu(self.fc1(x))                 # Fully Connected Layer
        x = self.fc2(x)                         # Output Layer
        return x
