import torch.nn as nn
import torch
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 16, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 16, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        # Compute queries, keys, and values
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # (B, N, C)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # (B, C, N)
        energy = torch.bmm(proj_query, proj_key)  # (B, N, N) batch matrix multiplication
        attention = self.softmax(energy)  # Attention map (B, N, N)

        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # (B, C, N)

        # Apply attention to values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B, C, N)
        out = out.view(batch_size, C, width, height)  # Reshape to original dimensions

        # Apply scaling factor gamma
        out = self.gamma * out + x
        return out, attention
