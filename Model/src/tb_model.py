import torch.nn as nn
import torch.nn.functional as F

class TB_CNN(nn.Module):
    def __init__(self):
        super(TB_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(8 * 56 * 56, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 8 * 56 * 56)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
