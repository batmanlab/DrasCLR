import torch
import torch.nn as nn
from models.condconv3d import CondConv3D as Conv3d

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts, num_coordinates):
        super(Block, self).__init__()
        self.num_experts = num_experts
        self.num_coordinates = num_coordinates
        self.conv1 = Conv3d(in_channels, out_channels, kernel_size=3, padding=1, num_experts=self.num_experts, num_coordinates=self.num_coordinates)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = Conv3d(out_channels, out_channels, kernel_size=3, padding=1, num_experts=self.num_experts, num_coordinates=self.num_coordinates)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, num_experts=self.num_experts, num_coordinates=self.num_coordinates)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.act = nn.ELU(inplace=True)

    def forward(self, x, loc):
        x = self.conv1(x, loc)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x, loc)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv3(x, loc)
        x = self.bn3(x)
        x = self.act(x)
        return x


class Encoder(nn.Module):

    def __init__(self, rep_dim, moco_dim, num_experts, num_coordinates):
        super(Encoder, self).__init__()
        self.rep_dim = rep_dim
        self.moco_dim = moco_dim
        self.num_experts = num_experts
        self.num_coordinates = num_coordinates
        self.conv1 = Conv3d(1, 8, kernel_size=3, stride=1, padding=1, num_experts=self.num_experts, num_coordinates=self.num_coordinates)
        self.bn1 = nn.BatchNorm3d(8)
        self.act = nn.ELU()
        self.conv2 = Conv3d(8, 8, kernel_size=3, stride=2, padding=1, num_experts=self.num_experts, num_coordinates=self.num_coordinates)
        self.bn2 = nn.BatchNorm3d(8)
        self.downsample1 = Block(8, 16, self.num_experts, self.num_coordinates)
        self.downsample2 = Block(16, 32, self.num_experts, self.num_coordinates)
        self.downsample3 = Block(32, 64, self.num_experts, self.num_coordinates)
        self.conv3 = Conv3d(64, 128, kernel_size=3, stride=1, padding=1, num_experts=self.num_experts, num_coordinates=self.num_coordinates)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = Conv3d(128, rep_dim, kernel_size=3, stride=2, padding=1, num_experts=self.num_experts, num_coordinates=self.num_coordinates)
        self.bn4 = nn.BatchNorm3d(rep_dim)
        self.fc = nn.Linear(rep_dim, moco_dim)

    def forward(self, x, loc):
        x = self.conv1(x, loc)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x, loc)
        x = self.bn2(x)
        x = self.act(x)
        x = self.downsample1(x, loc)
        x = self.downsample2(x, loc)
        x = self.downsample3(x, loc)
        x = self.conv3(x, loc)
        x = self.bn3(x)
        x = self.act(x)
        x = self.conv4(x, loc)
        x = self.bn4(x)
        x = self.act(x)
        h = torch.flatten(x, 1)
        z = self.fc(h)
        return z, h