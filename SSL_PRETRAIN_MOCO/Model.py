import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# DenseNet-B
class _Bottleneck(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool):
        super(_Bottleneck, self).__init__()
        interChannels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(n_channels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(
            interChannels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# single layer
class _SingleLayer(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool):
        super(_SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(
            n_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# transition layer
class _Transition(nn.Module):
    def __init__(self, n_channels: int, n_out_channels: int, use_dropout: bool):
        super(_Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_out_channels)
        self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int,
        num_layers: int,
        reduction: float = 0.5,
        bottleneck: bool = True,
        use_dropout: bool = True,
    ):
        super(DenseNet, self).__init__()
        n_dense_blocks = num_layers
        n_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(
            1, n_channels, kernel_size=7, padding=3, stride=2, bias=False
        )
        self.norm1 = nn.BatchNorm2d(n_channels)
        self.dense1 = self._make_dense(
            n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout
        )
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans1 = _Transition(n_channels, n_out_channels, use_dropout)

        n_channels = n_out_channels
        self.dense2 = self._make_dense(
            n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout
        )
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = _Transition(n_channels, n_out_channels, use_dropout)

        n_channels = n_out_channels
        self.dense3 = self._make_dense(
            n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout
        )

        self.out_channels = n_channels + n_dense_blocks * growth_rate
        self.post_norm = nn.BatchNorm2d(self.out_channels)

    @staticmethod
    def _make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout):
        layers = []
        for _ in range(int(n_dense_blocks)):
            if bottleneck:
                layers.append(_Bottleneck(n_channels, growth_rate, use_dropout))
            else:
                layers.append(_SingleLayer(n_channels, growth_rate, use_dropout))
            n_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, x_mask):
        out = self.conv1(x)
        out_first_layer = self.norm1(out)  # First layer feature map
        out = F.relu(out_first_layer, inplace=True)
        out_mask = x_mask[:, 0::2, 0::2]
        out = F.max_pool2d(out, 2, ceil_mode=True)
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense1(out)
        out = self.trans1(out)
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense2(out)
        out = self.trans2(out)
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense3(out)
        out = self.post_norm(out)
        return out, out_first_layer, out_mask


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super(ProjectionHead, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Adjust the in_features of the first fully connected layer
        self.fc1 = nn.Linear(in_dim, 1024)  # Changed from 684 to in_dim
        self.bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, out_dim)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor (batch_size, in_dim)
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
class Model(nn.Module):
    def __init__(self, growth_rate: int, num_layers: int, reduction: float = 0.5, bottleneck: bool = True, use_dropout: bool = True):
        super(Model, self).__init__()
        self.backbone = DenseNet(growth_rate, num_layers, reduction, bottleneck, use_dropout)
        self.projection_head = ProjectionHead(self.backbone.out_channels, out_dim=512)

    def forward(self, x, x_mask):
        x, x_first_layer, x_mask = self.backbone(x, x_mask)
        x_proj = self.projection_head(x)
        return x, x_proj, x_first_layer, x_mask