import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.z_size = config["z_size"]
        self.point_dim = config["point_dim"]
        self.gf_dim = config["gf_dim"]
        self.linear_1 = nn.Linear(
            self.z_size + self.point_dim, self.gf_dim * 8, bias=True
        )
        self.linear_2 = nn.Linear(self.gf_dim * 8, self.gf_dim * 8, bias=True)
        self.linear_3 = nn.Linear(self.gf_dim * 8, self.gf_dim * 8, bias=True)
        self.linear_4 = nn.Linear(self.gf_dim * 8, self.gf_dim * 4, bias=True)
        self.linear_5 = nn.Linear(self.gf_dim * 4, self.gf_dim * 2, bias=True)
        self.linear_6 = nn.Linear(self.gf_dim * 2, self.gf_dim * 1, bias=True)
        self.linear_7 = nn.Linear(self.gf_dim * 1, 1, bias=True)
        nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_1.bias, 0)
        nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_2.bias, 0)
        nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_3.bias, 0)
        nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_4.bias, 0)
        nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_5.bias, 0)
        nn.init.normal_(self.linear_6.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_6.bias, 0)
        nn.init.normal_(self.linear_7.weight, mean=1e-5, std=0.02)
        nn.init.constant_(self.linear_7.bias, 0)

    def forward(self, points, z, is_training=False):
        zs = z.view(-1, 1, self.z_size).repeat(1, points.size()[1], 1)
        pointz = torch.cat([points, zs], 2)

        l1 = self.linear_1(pointz)
        l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

        l4 = self.linear_4(l3)
        l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

        l5 = self.linear_5(l4)
        l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

        l6 = self.linear_6(l5)
        l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

        l7 = self.linear_7(l6)

        l7 = torch.max(torch.min(l7, l7 * 0.01 + 0.99), l7 * 0.01)

        return l7


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.ef_dim = config["ef_dim"]
        self.z_size = config["z_size"]
        self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=False)
        self.in_1 = nn.InstanceNorm3d(self.ef_dim)
        self.conv_2 = nn.Conv3d(
            self.ef_dim, self.ef_dim * 2, 4, stride=2, padding=1, bias=False
        )
        self.in_2 = nn.InstanceNorm3d(self.ef_dim * 2)
        self.conv_3 = nn.Conv3d(
            self.ef_dim * 2, self.ef_dim * 4, 4, stride=2, padding=1, bias=False
        )
        self.in_3 = nn.InstanceNorm3d(self.ef_dim * 4)
        self.conv_4 = nn.Conv3d(
            self.ef_dim * 4, self.ef_dim * 8, 4, stride=2, padding=1, bias=False
        )
        self.in_4 = nn.InstanceNorm3d(self.ef_dim * 8)
        self.conv_5 = nn.Conv3d(
            self.ef_dim * 8, self.z_size, 4, stride=1, padding=0, bias=True
        )
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.xavier_uniform_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias, 0)

    def forward(self, inputs, is_training=False):
        d_1 = self.in_1(self.conv_1(inputs))
        d_1 = F.leaky_relu(d_1, negative_slope=0.02, inplace=True)

        d_2 = self.in_2(self.conv_2(d_1))
        d_2 = F.leaky_relu(d_2, negative_slope=0.02, inplace=True)

        d_3 = self.in_3(self.conv_3(d_2))
        d_3 = F.leaky_relu(d_3, negative_slope=0.02, inplace=True)

        d_4 = self.in_4(self.conv_4(d_3))
        d_4 = F.leaky_relu(d_4, negative_slope=0.02, inplace=True)

        d_5 = self.conv_5(d_4)
        d_5 = d_5.view(-1, self.z_size)

        return torch.sigmoid(d_5)
