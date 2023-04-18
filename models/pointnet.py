import torch
from torch import nn
from torch.nn import functional as F


class PointNet(nn.Module):
    def __init__(self, lat_size, conv_size):
        super(PointNet, self).__init__()
        conv_layers = []
        for i in range(len(conv_size)-1):
            conv_layers.append(nn.Conv1d(conv_size[i], conv_size[i+1], 1))
            conv_layers.append(nn.BatchNorm1d(conv_size[i+1]))
            conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv1d(conv_size[-1], lat_size, 1))
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x, R=None, A=None):
        if R is not None:
            x = torch.matmul(x, R)
        if A is not None:
            x = torch.matmul(x, A[:, :, :3]) + A[:, :, 3].unsqueeze(1)
        x = x.permute(0, 2, 1)
        assert x.size(1) == 3, "Wrong input dimension"
        x = self.conv_layers(x)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x


class FCDecoder(nn.Module):
    def __init__(self, lat_size, fc_size):
        super(FCDecoder, self).__init__()
        fc_size = [lat_size] + list(fc_size)
        fc_layers = []
        for i in range(len(fc_size)-1):
            fc_layers.append(nn.Linear(fc_size[i], fc_size[i+1]))
            if i != len(fc_size)-2:
                fc_layers.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x, R=None, A=None):
        x = self.fc_layers(x)
        x = x.view(x.size(0), -1, 3)
        if R is not None:
            x = torch.matmul(x, R.permute(0, 2, 1))
        if A is not None:
            x = torch.matmul(x - A[:, :, 3].unsqueeze(1), torch.inverse(A[:, :, :3]))
        return x

