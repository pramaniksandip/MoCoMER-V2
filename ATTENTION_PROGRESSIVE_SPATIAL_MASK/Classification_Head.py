import torch.nn as nn
import torch.nn.functional as F

class Classification_Head(nn.Module):
    def __init__(self, in_dim, out_dim=110):
        super(Classification_Head, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_dim, 1024)  # Changed from 684 to in_dim
        self.fc2 = nn.Linear(1024, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor (batch_size, in_dim)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x