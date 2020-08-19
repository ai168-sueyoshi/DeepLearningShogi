﻿import torch
import torch.nn as nn
import torch.nn.functional as F

from dlshogi.common import *

class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias=nn.Parameter(torch.Tensor(shape))

    def forward(self, input):
        return input + self.bias

k = 192
fcl = 256 # fully connected layers
class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super(PolicyValueNetwork, self).__init__()
        self.l1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=1, padding=0, bias=False)
        self.l1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=k, kernel_size=1, bias=False) # pieces_in_hand
        self.l2 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l3 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l4 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l5 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l6 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l7 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l8 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l9 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l10 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l11 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l12 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l13 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l14 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l15 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l16 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l17 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l18 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l19 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l20 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l21 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # policy network
        self.l22 = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.l22_2 = Bias(9*9*MAX_MOVE_LABEL_NUM)
        # value network
        self.l22_v = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1)
        self.l23_v = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        self.l24_v = nn.Linear(fcl, 1)
        self.norm1 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm2 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm3 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm4 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm5 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm6 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm7 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm8 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm9 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm10 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm11 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm12 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm13 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm14 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm15 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm16 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm17 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm18 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm19 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm20 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm21 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm22_v = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM, eps=2e-05)

    def __call__(self, x1, x2):
        u1_1_1 = self.l1_1_1(x1)
        u1_1_2 = self.l1_1_2(x1)
        u1_2 = self.l1_2(x2)
        u1 = F.relu(self.norm1(u1_1_1 + u1_1_2 + u1_2))
        # Residual block
        h2 = F.relu(self.norm2(self.l2(u1)))
        h3 = self.norm3(self.l3(h2))
        u3 = F.relu(h3 + u1)
        # Residual block
        h4 = F.relu(self.norm4(self.l4(u3)))
        h5 = self.norm5(self.l5(h4))
        u5 = F.relu(h5 + u3)
        # Residual block
        h6 = F.relu(self.norm6(self.l6(u5)))
        h7 = self.norm7(self.l7(h6))
        u7 = F.relu(h7 + u5)
        # Residual block
        h8 = F.relu(self.norm8(self.l8(u7)))
        h9 = self.norm9(self.l9(h8))
        u9 = F.relu(h9 + u7)
        # Residual block
        h10 = F.relu(self.norm10(self.l10(u9)))
        h11 = self.norm11(self.l11(h10))
        u11 = F.relu(h11 + u9)
        # Residual block
        h12 = F.relu(self.norm12(self.l12(u11)))
        h13 = self.norm13(self.l13(h12))
        u13 = F.relu(h13 + u11)
        # Residual block
        h14 = F.relu(self.norm14(self.l14(u13)))
        h15 = self.norm15(self.l15(h14))
        u15 = F.relu(h15 + u13)
        # Residual block
        h16 = F.relu(self.norm16(self.l16(u15)))
        h17 = self.norm17(self.l17(h16))
        u17 = F.relu(h17 + u15)
        # Residual block
        h18 = F.relu(self.norm18(self.l18(u17)))
        h19 = self.norm19(self.l19(h18))
        u19 = F.relu(h19 + u17)
        # Residual block
        h20 = F.relu(self.norm20(self.l20(u19)))
        h21 = self.norm21(self.l21(h20))
        u21 = F.relu(h21 + u19)
        # policy network
        h22 = self.l22(u21)
        h22_1 = self.l22_2(h22.view(-1, 9*9*MAX_MOVE_LABEL_NUM))
        # value network
        h22_v = F.relu(self.norm22_v(self.l22_v(u21)))
        h23_v = F.relu(self.l23_v(h22_v.view(-1, 9*9*MAX_MOVE_LABEL_NUM)))
        return h22_1, self.l24_v(h23_v)
