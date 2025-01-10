import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLocationModel(nn.Module):
    def __init__(self):
        super(CNNLocationModel, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)   # Conv (1 -> 64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Conv (64 -> 128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # Conv (128 -> 256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  # Conv (256 -> 512)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(512 * 2 * 2, 512)  # Fully connected layer 1
        self.fc2 = nn.Linear(512, 2)  # Final output (ΔLat, ΔLon)
        
    def forward(self, x):
        # Convolutional and Pooling Layers
        x = F.relu(self.conv1(x))  # (batch, 64, 32, 32)
        x = F.max_pool2d(x, 2)     # (batch, 64, 16, 16)
        
        x = F.relu(self.conv2(x))  # (batch, 128, 16, 16)
        x = F.max_pool2d(x, 2)     # (batch, 128, 8, 8)
        
        x = F.relu(self.conv3(x))  # (batch, 256, 8, 8)
        x = F.max_pool2d(x, 2)     # (batch, 256, 4, 4)
        
        x = F.relu(self.conv4(x))  # (batch, 512, 4, 4)
        x = F.max_pool2d(x, 2)     # (batch, 512, 2, 2)
        
        # Flatten and FC Layers
        x = x.view(x.size(0), -1)  # Flatten (batch, 512*2*2)
        x = F.relu(self.fc1(x))    # (batch, 512)
        x = self.fc2(x)            # (batch, 2)
        
        return x