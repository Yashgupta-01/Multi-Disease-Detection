import torch.nn as nn
import torch.nn.functional as F

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
        x = self.pool(F.relu(self.conv1(x)))  # -> (16, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # -> (32, 56, 56)
        x = x.view(-1, 32 * 56 * 56)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
