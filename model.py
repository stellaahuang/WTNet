import torch
import torch.nn as nn
import logging
import sys
from torch.nn import functional as F
import math
from thop import profile

class Encoder_A_B_M(nn.Module):
    def __init__(self, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6):
        super(Encoder_A_B_M, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.number_en_1 = 2
        self.number_en_2 = 2
        self.number_en_3 = 2
        self.number_en_4 = 2
        self.number_en_5 = 2
        self.number_en_6 = 2

        # Layer 1
        self.en_1_input = nn.Sequential(
            nn.Conv2d(3, dim_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_1),
            self.activation)
        self.en_1_res = nn.ModuleList()
        for i in range(self.number_en_1):
            self.en_1_res.append(nn.Sequential(
                nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_1),
                self.activation,
                nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_1),
                self.activation))

        # Layer 2
        self.en_2_input = nn.Sequential(
            nn.Conv2d(dim_1, dim_2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim_2),
            self.activation)
        self.en_2_res = nn.ModuleList()
        for i in range(self.number_en_2):
            self.en_2_res.append(nn.Sequential(
                nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_2),
                self.activation,
                nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_2),
                self.activation))

        # Layer 3
        self.en_3_input = nn.Sequential(
            nn.Conv2d(dim_2, dim_3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim_3),
            self.activation)
        self.en_3_res = nn.ModuleList()
        for i in range(self.number_en_3):
            self.en_3_res.append(nn.Sequential(
                nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_3),
                self.activation,
                nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_3),
                self.activation))

        # Layer 4
        self.en_4_input = nn.Sequential(
            nn.Conv2d(dim_3, dim_4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim_4),
            self.activation)
        self.en_4_res = nn.ModuleList()
        for i in range(self.number_en_4):
            self.en_4_res.append(nn.Sequential(
                nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_4),
                self.activation,
                nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_4),
                self.activation))

        # Layer 5
        self.en_5_input = nn.Sequential(
            nn.Conv2d(dim_4, dim_5, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim_5),
            self.activation)
        self.en_5_res = nn.ModuleList()
        for i in range(self.number_en_5):
            self.en_5_res.append(nn.Sequential(
                nn.Conv2d(dim_5, dim_5, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_5),
                self.activation,
                nn.Conv2d(dim_5, dim_5, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_5),
                self.activation))

        # Layer 6
        self.en_6_input = nn.Sequential(
            nn.Conv2d(dim_5, dim_6, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim_6),
            self.activation)
        self.en_6_res = nn.ModuleList()
        for i in range(self.number_en_6):
            self.en_6_res.append(nn.Sequential(
                nn.Conv2d(dim_6, dim_6, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_6),
                self.activation,
                nn.Conv2d(dim_6, dim_6, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_6),
                self.activation))

    def forward(self, x):
        hx = self.en_1_input(x)
        for i in range(self.number_en_1):
            hx = self.activation(self.en_1_res[i](hx) + hx)
        res_1 = hx

        hx = self.en_2_input(hx)
        for i in range(self.number_en_2):
            hx = self.activation(self.en_2_res[i](hx) + hx)
        res_2 = hx

        hx = self.en_3_input(hx)
        for i in range(self.number_en_3):
            hx = self.activation(self.en_3_res[i](hx) + hx)
        res_3 = hx

        hx = self.en_4_input(hx)
        for i in range(self.number_en_4):
            hx = self.activation(self.en_4_res[i](hx) + hx)
        res_4 = hx

        hx = self.en_5_input(hx)
        for i in range(self.number_en_5):
            hx = self.activation(self.en_5_res[i](hx) + hx)
        res_5 = hx

        hx = self.en_6_input(hx)
        for i in range(self.number_en_6):
            hx = self.activation(self.en_6_res[i](hx) + hx)

        return hx, res_1, res_2, res_3, res_4, res_5


class Decoder_A(nn.Module):
    def __init__(self, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6):
        super(Decoder_A, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.number_de_1 = 2
        self.number_de_2 = 2
        self.number_de_3 = 2
        self.number_de_4 = 2
        self.number_de_5 = 2

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder 5
        self.de_5_fuse = nn.Sequential(
            nn.Conv2d(dim_6 + dim_5, dim_5, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_5),
            self.activation)
        self.de_5_res = nn.ModuleList()
        for i in range(self.number_de_5):
            self.de_5_res.append(nn.Sequential(
                nn.Conv2d(dim_5, dim_5, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_5),
                self.activation,
                nn.Conv2d(dim_5, dim_5, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_5),
                self.activation))

        # Decoder 4
        self.de_4_fuse = nn.Sequential(
            nn.Conv2d(dim_5 + dim_4, dim_4, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_4),
            self.activation)
        self.de_4_res = nn.ModuleList()
        for i in range(self.number_de_4):
            self.de_4_res.append(nn.Sequential(
                nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_4),
                self.activation,
                nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_4),
                self.activation))

        # Decoder 3
        self.de_3_fuse = nn.Sequential(
            nn.Conv2d(dim_4 + dim_3, dim_3, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_3),
            self.activation)
        self.de_3_res = nn.ModuleList()
        for i in range(self.number_de_3):
            self.de_3_res.append(nn.Sequential(
                nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_3),
                self.activation,
                nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_3),
                self.activation))

        # Decoder 2
        self.de_2_fuse = nn.Sequential(
            nn.Conv2d(dim_3 + dim_2, dim_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_2),
            self.activation)
        self.de_2_res = nn.ModuleList()
        for i in range(self.number_de_2):
            self.de_2_res.append(nn.Sequential(
                nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_2),
                self.activation,
                nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_2),
                self.activation))

        # Decoder 1
        self.de_1_fuse = nn.Sequential(
            nn.Conv2d(dim_2 + dim_1, dim_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_1),
            self.activation)
        self.de_1_res = nn.ModuleList()
        for i in range(self.number_de_1):
            self.de_1_res.append(nn.Sequential(
                nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_1),
                self.activation,
                nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_1),
                self.activation))

        self.output = nn.Sequential(
            nn.Conv2d(dim_1, 1, kernel_size=3, padding=1),
            nn.ReLU())

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, res_1, res_2, res_3, res_4, res_5):
        hx = self.up(x)
        hx = self.de_5_fuse(torch.cat((hx, res_5), dim=1))
        for i in range(self.number_de_5):
            hx = self.activation(self.de_5_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_4_fuse(torch.cat((hx, res_4), dim=1))
        for i in range(self.number_de_4):
            hx = self.activation(self.de_4_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_3_fuse(torch.cat((hx, res_3), dim=1))
        for i in range(self.number_de_3):
            hx = self.activation(self.de_3_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_2_fuse(torch.cat((hx, res_2), dim=1))
        for i in range(self.number_de_2):
            hx = self.activation(self.de_2_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_1_fuse(torch.cat((hx, res_1), dim=1))
        for i in range(self.number_de_1):
            hx = self.activation(self.de_1_res[i](hx) + hx)

        hx = torch.exp(-self.pool(self.output(hx)))
        # hx = torch.exp(-self.output(hx))  # A map

        return hx


class Decoder_B(nn.Module):
    def __init__(self, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6):
        super(Decoder_B, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.number_de_1 = 2
        self.number_de_2 = 2
        self.number_de_3 = 2
        self.number_de_4 = 2
        self.number_de_5 = 2

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder 5
        self.de_5_fuse = nn.Sequential(
            nn.Conv2d(dim_6 + dim_5, dim_5, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_5),
            self.activation)
        self.de_5_res = nn.ModuleList()
        for i in range(self.number_de_5):
            self.de_5_res.append(nn.Sequential(
                nn.Conv2d(dim_5, dim_5, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_5),
                self.activation,
                nn.Conv2d(dim_5, dim_5, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_5),
                self.activation))

        # Decoder 4
        self.de_4_fuse = nn.Sequential(
            nn.Conv2d(dim_5 + dim_4, dim_4, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_4),
            self.activation)
        self.de_4_res = nn.ModuleList()
        for i in range(self.number_de_4):
            self.de_4_res.append(nn.Sequential(
                nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_4),
                self.activation,
                nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_4),
                self.activation))

        # Decoder 3
        self.de_3_fuse = nn.Sequential(
            nn.Conv2d(dim_4 + dim_3, dim_3, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_3),
            self.activation)
        self.de_3_res = nn.ModuleList()
        for i in range(self.number_de_3):
            self.de_3_res.append(nn.Sequential(
                nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_3),
                self.activation,
                nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_3),
                self.activation))

        # Decoder 2
        self.de_2_fuse = nn.Sequential(
            nn.Conv2d(dim_3 + dim_2, dim_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_2),
            self.activation)
        self.de_2_res = nn.ModuleList()
        for i in range(self.number_de_2):
            self.de_2_res.append(nn.Sequential(
                nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_2),
                self.activation,
                nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_2),
                self.activation))

        # Decoder 1
        self.de_1_fuse = nn.Sequential(
            nn.Conv2d(dim_2 + dim_1, dim_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_1),
            self.activation)
        self.de_1_res = nn.ModuleList()
        for i in range(self.number_de_1):
            self.de_1_res.append(nn.Sequential(
                nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_1),
                self.activation,
                nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_1),
                self.activation))

        self.output = nn.Sequential(
            nn.Conv2d(dim_1, 1, kernel_size=3, padding=1),
            nn.ReLU())

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, res_1, res_2, res_3, res_4, res_5):
        hx = self.up(x)
        hx = self.de_5_fuse(torch.cat((hx, res_5), dim=1))
        for i in range(self.number_de_5):
            hx = self.activation(self.de_5_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_4_fuse(torch.cat((hx, res_4), dim=1))
        for i in range(self.number_de_4):
            hx = self.activation(self.de_4_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_3_fuse(torch.cat((hx, res_3), dim=1))
        for i in range(self.number_de_3):
            hx = self.activation(self.de_3_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_2_fuse(torch.cat((hx, res_2), dim=1))
        for i in range(self.number_de_2):
            hx = self.activation(self.de_2_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_1_fuse(torch.cat((hx, res_1), dim=1))
        for i in range(self.number_de_1):
            hx = self.activation(self.de_1_res[i](hx) + hx)

        hx = torch.exp(-self.pool(self.output(hx)))
        # hx = torch.exp(-self.output(hx))  # beta map

        return hx


class Decoder_M(nn.Module):
    def __init__(self, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6):
        super(Decoder_M, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.number_de_1 = 2
        self.number_de_2 = 2
        self.number_de_3 = 2
        self.number_de_4 = 2
        self.number_de_5 = 2

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder 5
        self.de_5_fuse = nn.Sequential(
            nn.Conv2d(dim_6 + dim_5, dim_5, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_5),
            self.activation)
        self.de_5_res = nn.ModuleList()
        for i in range(self.number_de_5):
            self.de_5_res.append(nn.Sequential(
                nn.Conv2d(dim_5, dim_5, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_5),
                self.activation,
                nn.Conv2d(dim_5, dim_5, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_5),
                self.activation))

        # Decoder 4
        self.de_4_fuse = nn.Sequential(
            nn.Conv2d(dim_5 + dim_4, dim_4, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_4),
            self.activation)
        self.de_4_res = nn.ModuleList()
        for i in range(self.number_de_4):
            self.de_4_res.append(nn.Sequential(
                nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_4),
                self.activation,
                nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_4),
                self.activation))

        # Decoder 3
        self.de_3_fuse = nn.Sequential(
            nn.Conv2d(dim_4 + dim_3, dim_3, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_3),
            self.activation)
        self.de_3_res = nn.ModuleList()
        for i in range(self.number_de_3):
            self.de_3_res.append(nn.Sequential(
                nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_3),
                self.activation,
                nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_3),
                self.activation))

        # Decoder 2
        self.de_2_fuse = nn.Sequential(
            nn.Conv2d(dim_3 + dim_2, dim_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_2),
            self.activation)
        self.de_2_res = nn.ModuleList()
        for i in range(self.number_de_2):
            self.de_2_res.append(nn.Sequential(
                nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_2),
                self.activation,
                nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_2),
                self.activation))

        # Decoder 1
        self.de_1_fuse = nn.Sequential(
            nn.Conv2d(dim_2 + dim_1, dim_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_1),
            self.activation)
        self.de_1_res = nn.ModuleList()
        for i in range(self.number_de_1):
            self.de_1_res.append(nn.Sequential(
                nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_1),
                self.activation,
                nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_1),
                self.activation))

        # Output Layer
        self.output = nn.Sequential(
            nn.Conv2d(dim_1, 1, kernel_size=3, padding=1),
            nn.ReLU())

    def forward(self, x, res_1, res_2, res_3, res_4, res_5):
        hx = self.up(x)
        hx = self.de_5_fuse(torch.cat((hx, res_5), dim=1))
        for i in range(self.number_de_5):
            hx = self.activation(self.de_5_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_4_fuse(torch.cat((hx, res_4), dim=1))
        for i in range(self.number_de_4):
            hx = self.activation(self.de_4_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_3_fuse(torch.cat((hx, res_3), dim=1))
        for i in range(self.number_de_3):
            hx = self.activation(self.de_3_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_2_fuse(torch.cat((hx, res_2), dim=1))
        for i in range(self.number_de_2):
            hx = self.activation(self.de_2_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_1_fuse(torch.cat((hx, res_1), dim=1))
        for i in range(self.number_de_1):
            hx = self.activation(self.de_1_res[i](hx) + hx)

        hx = torch.exp(-self.output(hx))

        return hx



# class Decoder_T(nn.Module):
#     def __init__(self, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6):
#         super(Decoder_T, self).__init__()

#         self.activation = nn.LeakyReLU(0.2, True)

#         self.number_de_1 = 2
#         self.number_de_2 = 2
#         self.number_de_3 = 2
#         self.number_de_4 = 2
#         self.number_de_5 = 2

#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.de_5_fuse = nn.Sequential(
#             nn.Conv2d(dim_6 + dim_5, dim_5, kernel_size=3, padding=1),
#             self.activation)
#         self.de_5_res = nn.ModuleList()
#         for i in range(self.number_de_5):
#             self.de_5_res.append(nn.Sequential(
#             nn.Conv2d(dim_5, dim_5, kernel_size=3, padding=1),
#             self.activation,
#             nn.Conv2d(dim_5, dim_5, kernel_size=3, padding=1),
#             self.activation))

#         self.de_4_fuse = nn.Sequential(
#             nn.Conv2d(dim_5 + dim_4, dim_4, kernel_size=3, padding=1),
#             self.activation)
#         self.de_4_res = nn.ModuleList()
#         for i in range(self.number_de_4):
#             self.de_4_res.append(nn.Sequential(
#             nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
#             self.activation,
#             nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
#             self.activation))

#         self.de_3_fuse = nn.Sequential(
#             nn.Conv2d(dim_4 + dim_3, dim_3, kernel_size=3, padding=1),
#             self.activation)
#         self.de_3_res = nn.ModuleList()
#         for i in range(self.number_de_3):
#             self.de_3_res.append(nn.Sequential(
#             nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
#             self.activation,
#             nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
#             self.activation))

#         self.de_2_fuse = nn.Sequential(
#             nn.Conv2d(dim_3 + dim_2, dim_2, kernel_size=3, padding=1),
#             self.activation)
#         self.de_2_res = nn.ModuleList()
#         for i in range(self.number_de_2):
#             self.de_2_res.append(nn.Sequential(
#             nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
#             self.activation,
#             nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
#             self.activation))

#         self.de_1_fuse = nn.Sequential(
#             nn.Conv2d(dim_2 + dim_1, dim_1, kernel_size=3, padding=1),
#             self.activation)
#         self.de_1_res = nn.ModuleList()
#         for i in range(self.number_de_1):
#             self.de_1_res.append(nn.Sequential(
#             nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
#             self.activation,
#             nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
#             self.activation))

#         self.output = nn.Sequential(
#             nn.Conv2d(dim_1, 3, kernel_size=3, padding=1),
#             nn.ReLU())

#     def forward(self, x, res_1, res_2, res_3, res_4, res_5):

#         hx = self.up(x)
#         hx = self.de_5_fuse(torch.cat((hx, res_5), dim=1))
#         for i in range(self.number_de_5):
#             hx = self.activation(self.de_5_res[i](hx) + hx)

#         hx = self.up(hx)
#         hx = self.de_4_fuse(torch.cat((hx, res_4), dim=1))
#         for i in range(self.number_de_4):
#             hx = self.activation(self.de_4_res[i](hx) + hx)

#         hx = self.up(hx)
#         hx = self.de_3_fuse(torch.cat((hx, res_3), dim=1))
#         for i in range(self.number_de_3):
#             hx = self.activation(self.de_3_res[i](hx) + hx)

#         hx = self.up(hx)
#         hx = self.de_2_fuse(torch.cat((hx, res_2), dim=1))
#         for i in range(self.number_de_2):
#             hx = self.activation(self.de_2_res[i](hx) + hx)

#         hx = self.up(hx)
#         hx = self.de_1_fuse(torch.cat((hx, res_1), dim=1))
#         for i in range(self.number_de_1):
#             hx = self.activation(self.de_1_res[i](hx) + hx)

#         hx = torch.exp(-self.output(hx))

#         return hx


class RefineNet(nn.Module):
    def __init__(self, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6):
        super(RefineNet, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.number_en_1 = 2
        self.number_en_2 = 2
        self.number_en_3 = 2
        self.number_en_4 = 2
        self.number_en_5 = 2
        self.number_en_6 = 2
        self.number_de_1 = 2
        self.number_de_2 = 2
        self.number_de_3 = 2
        self.number_de_4 = 2
        self.number_de_5 = 2

        # Encoder 1
        self.en_1_input = nn.Sequential(
            nn.Conv2d(6, dim_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_1),
            self.activation)
        self.en_1_res = nn.ModuleList()
        for i in range(self.number_en_1):
            self.en_1_res.append(nn.Sequential(
                nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_1),
                self.activation,
                nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_1),
                self.activation))

        # Encoder 2
        self.en_2_input = nn.Sequential(
            nn.Conv2d(dim_1, dim_2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim_2),
            self.activation)
        self.en_2_res = nn.ModuleList()
        for i in range(self.number_en_2):
            self.en_2_res.append(nn.Sequential(
                nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_2),
                self.activation,
                nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_2),
                self.activation))

        # Encoder 3
        self.en_3_input = nn.Sequential(
            nn.Conv2d(dim_2, dim_3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim_3),
            self.activation)
        self.en_3_res = nn.ModuleList()
        for i in range(self.number_en_3):
            self.en_3_res.append(nn.Sequential(
                nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_3),
                self.activation,
                nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_3),
                self.activation))

        # Encoder 4
        self.en_4_input = nn.Sequential(
            nn.Conv2d(dim_3, dim_4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim_4),
            self.activation)
        self.en_4_res = nn.ModuleList()
        for i in range(self.number_en_4):
            self.en_4_res.append(nn.Sequential(
                nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_4),
                self.activation,
                nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_4),
                self.activation))

        # Encoder 5
        self.en_5_input = nn.Sequential(
            nn.Conv2d(dim_4, dim_5, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim_5),
            self.activation)
        self.en_5_res = nn.ModuleList()
        for i in range(self.number_en_5):
            self.en_5_res.append(nn.Sequential(
                nn.Conv2d(dim_5, dim_5, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_5),
                self.activation,
                nn.Conv2d(dim_5, dim_5, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_5),
                self.activation))

        # Encoder 6
        self.en_6_input = nn.Sequential(
            nn.Conv2d(dim_5, dim_6, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim_6),
            self.activation)
        self.en_6_res = nn.ModuleList()
        for i in range(self.number_en_6):
            self.en_6_res.append(nn.Sequential(
                nn.Conv2d(dim_6, dim_6, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_6),
                self.activation,
                nn.Conv2d(dim_6, dim_6, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_6),
                self.activation))

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder 5
        self.de_5_fuse = nn.Sequential(
            nn.Conv2d(dim_6 + dim_5, dim_5, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_5),
            self.activation)
        self.de_5_res = nn.ModuleList()
        for i in range(self.number_de_5):
            self.de_5_res.append(nn.Sequential(
                nn.Conv2d(dim_5, dim_5, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_5),
                self.activation,
                nn.Conv2d(dim_5, dim_5, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_5),
                self.activation))

        # Decoder 4
        self.de_4_fuse = nn.Sequential(
            nn.Conv2d(dim_5 + dim_4, dim_4, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_4),
            self.activation)
        self.de_4_res = nn.ModuleList()
        for i in range(self.number_de_4):
            self.de_4_res.append(nn.Sequential(
                nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_4),
                self.activation,
                nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_4),
                self.activation))

        # Decoder 3
        self.de_3_fuse = nn.Sequential(
            nn.Conv2d(dim_4 + dim_3, dim_3, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_3),
            self.activation)
        self.de_3_res = nn.ModuleList()
        for i in range(self.number_de_3):
            self.de_3_res.append(nn.Sequential(
                nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_3),
                self.activation,
                nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_3),
                self.activation))

        # Decoder 2
        self.de_2_fuse = nn.Sequential(
            nn.Conv2d(dim_3 + dim_2, dim_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_2),
            self.activation)
        self.de_2_res = nn.ModuleList()
        for i in range(self.number_de_2):
            self.de_2_res.append(nn.Sequential(
                nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_2),
                self.activation,
                nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_2),
                self.activation))

        # Decoder 1
        self.de_1_fuse = nn.Sequential(
            nn.Conv2d(dim_2 + dim_1, dim_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_1),
            self.activation)
        self.de_1_res = nn.ModuleList()
        for i in range(self.number_de_1):
            self.de_1_res.append(nn.Sequential(
                nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_1),
                self.activation,
                nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_1),
                self.activation))

        # Output
        self.output = nn.Sequential(
            nn.Conv2d(dim_1, 3, kernel_size=3, padding=1),
            self.activation)

    def forward(self, x):
        hx = self.en_1_input(x)
        for i in range(self.number_en_1):
            hx = self.activation(self.en_1_res[i](hx) + hx)
        res_1 = hx

        hx = self.en_2_input(hx)
        for i in range(self.number_en_2):
            hx = self.activation(self.en_2_res[i](hx) + hx)
        res_2 = hx

        hx = self.en_3_input(hx)
        for i in range(self.number_en_3):
            hx = self.activation(self.en_3_res[i](hx) + hx)
        res_3 = hx

        hx = self.en_4_input(hx)
        for i in range(self.number_en_4):
            hx = self.activation(self.en_4_res[i](hx) + hx)
        res_4 = hx

        _, c, h, w = hx.shape
        hx = self.en_5_input(hx)
        for i in range(self.number_en_5):
            hx = self.activation(self.en_5_res[i](hx) + hx)
        res_5 = hx

        hx = self.en_6_input(hx)
        for i in range(self.number_en_6):
            hx = self.activation(self.en_6_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_5_fuse(torch.cat((hx, res_5), dim=1))
        for i in range(self.number_de_5):
            hx = self.activation(self.de_5_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_4_fuse(torch.cat((hx, res_4), dim=1))
        for i in range(self.number_de_4):
            hx = self.activation(self.de_4_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_3_fuse(torch.cat((hx, res_3), dim=1))
        for i in range(self.number_de_3):
            hx = self.activation(self.de_3_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_2_fuse(torch.cat((hx, res_2), dim=1))
        for i in range(self.number_de_2):
            hx = self.activation(self.de_2_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_1_fuse(torch.cat((hx, res_1), dim=1))
        for i in range(self.number_de_1):
            hx = self.activation(self.de_1_res[i](hx) + hx)

        hx = self.output(hx)

        return hx


class Network(nn.Module):
    def __init__(self, device, dim_1=8, dim_2=16, dim_3=32, dim_4=64, dim_5=128, dim_6=256):
        super(Network, self).__init__()
        self.device = device

        self.encoder_A_B_M = Encoder_A_B_M(dim_1, dim_2, dim_3, dim_4, dim_5, dim_6)
        self.decoder_A = Decoder_A(dim_1, dim_2, dim_3, dim_4, dim_5, dim_6)
        self.decoder_B = Decoder_B(dim_1, dim_2, dim_3, dim_4, dim_5, dim_6)
        self.decoder_M = Decoder_M(dim_1, dim_2, dim_3, dim_4, dim_5, dim_6)
        # self.decoder_T = Decoder_T(dim_1, dim_2, dim_3, dim_4, dim_5, dim_6)
        self.refinenet = RefineNet(dim_1, dim_2, dim_3, dim_4, dim_5, dim_6)

    def forward(self, deg, clean, depth):
        print('image resolution:')
        print(clean.shape)
        hx, res_1, res_2, res_3, res_4, res_5 = self.encoder_A_B_M(deg)
        print('feature resolution:')
        print(hx.shape)
        A =  self.decoder_A(hx, res_1, res_2, res_3, res_4, res_5)
        beta =  self.decoder_B(hx, res_1, res_2, res_3, res_4, res_5)
        mask =  self.decoder_M(hx, res_1, res_2, res_3, res_4, res_5)

        # formula
        T = torch.exp(-beta * depth)
        l = torch.randint(20, 60, (1,))
        S = (torch.rand(1) * 0.05 + 0.04 * l + 0.2).to(self.device)
        temp = torch.clamp(clean * (1 - mask) + S * mask, 0, 1)
        output = (temp * T + A * (1 - T)).clamp(0, 1)
        refine = (self.refinenet(torch.cat((output, clean), dim=1)) + output).clamp(0, 1)

        return refine, A, beta, mask

    # def forward(self, deg, clean, depth):
    #     hx, res_1, res_2, res_3, res_4, res_5 = self.encoder_A_B_M(deg)
    #     A =  self.decoder_A(hx, res_1, res_2, res_3, res_4, res_5)
    #     beta =  self.decoder_B(hx, res_1, res_2, res_3, res_4, res_5)
    #     mask =  self.decoder_M(hx, res_1, res_2, res_3, res_4, res_5)
    #     T =  self.decoder_T(hx, res_1, res_2, res_3, res_4, res_5)

    #     # formula
    #     T = torch.exp(-beta * depth)
    #     l = torch.randint(20, 60, (1,))
    #     S = (torch.rand(1) * 0.05 + 0.04 * l + 0.2).to(self.device)
    #     temp = torch.clamp(clean * (1 - mask) + S * mask, 0, 1)
    #     # output = (temp * T + A * (1 - T)).clamp(0, 1)
    #     output = (temp * T).clamp(0, 1)
    #     refine = (self.refinenet(torch.cat((output, clean), dim=1)) + output).clamp(0, 1)

    #     return refine, A, beta, mask, T

    def inference(self, clean, target, target_depth):
        hx, res_1, res_2, res_3, res_4, res_5 = self.encoder_A_B_M(target)
        A2 =  self.decoder_A(hx, res_1, res_2, res_3, res_4, res_5)
        beta2 =  self.decoder_B(hx, res_1, res_2, res_3, res_4, res_5)
        mask2 =  self.decoder_M(hx, res_1, res_2, res_3, res_4, res_5)

        # formula
        T2 = torch.exp(-beta2 * target_depth)
        l = torch.randint(20, 60, (1,))
        S = (torch.rand(1) * 0.05 + 0.04 * l + 0.2).to(self.device)
        temp = torch.clamp(clean * (1 - mask2) + S * mask2, 0, 1)
        output = (temp * T2 + A2 * (1 - T2)).clamp(0, 1)
        refine = (self.refinenet(torch.cat((output, clean), dim=1)) + output).clamp(0, 1)

        return refine, A2, beta2, mask2
