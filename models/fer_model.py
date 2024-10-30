import torch.nn as nn
import torch.nn.functional as F

class FERModel(nn.Module):
    def __init__(self, num_classes=7):  # Adjust num_classes based on your dataset
        super(FERModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # First conv layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Second conv layer
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # Fully connected layer (adjust input size)
        self.fc2 = nn.Linear(128, num_classes)  # Output layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # Max pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer
        return x
