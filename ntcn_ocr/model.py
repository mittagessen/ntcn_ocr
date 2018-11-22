# losely based on Temporal Convolutional Networks (TCN)

import torch

from torchvision import models
from torchvision import transforms
import torch.nn.functional as F
from torch import autograd

from torch import nn

from PIL import Image

from functools import partial

class CausalConv1d(nn.Conv1d):
    """
    Simple 1d causal convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.slice_padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(in_channels,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=self.slice_padding,
                                           dilation=dilation,
                                           groups=groups,
                                           bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        return result[:, :, :-self.slice_padding]

class CausalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0.1, reg='dropout'):
        super(CausalBlock, self).__init__()
        if reg == 'dropout2d':
            reg_l = partial(nn.Dropout2d, dropout)
        elif reg == 'dropout':
            reg_l = partial(nn.Dropout, dropout)
        elif reg == 'batchnorm':
            reg_l = partial(nn.BatchNorm1d, out_channels)
        else:
            raise Exception('invalid regularization layer selected')
        self.layer = nn.Sequential(CausalConv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation),
                                   nn.ReLU(),
                                   reg_l(),
                                   CausalConv1d(out_channels, out_channels, kernel_size, stride=stride, dilation=dilation),
                                   nn.ReLU(),
                                   reg_l(),
                                   CausalConv1d(out_channels, out_channels, kernel_size, stride=stride, dilation=dilation),
                                   reg_l())
        # downsampling for residual
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        o = self.layer(x)
        o = torch.relu(o + self.residual(x) if self.residual else o + x)
        return o


class CausalNet(nn.Module):

    def __init__(self, input_size, output_size, out_channels, layers, kernel_size, dropout=0.1, reg='dropout'):
        super(CausalNet, self).__init__()
        l = []
        l.append(CausalBlock(input_size, out_channels, kernel_size, stride=1, dilation=1, dropout=dropout))
        for i in range(layers):
            dilation_size = 2 ** (i+1)
            l.append(CausalBlock(out_channels, out_channels, kernel_size,
                                 stride=1, dilation=dilation_size,
                                 dropout=dropout, reg=reg))
        l.append(CausalBlock(out_channels, output_size, kernel_size, stride=1, dilation=(i+2**2), dropout=dropout, reg=reg))
        self.net = nn.Sequential(*l)
        self.init_weights()

    def forward(self, x):
        return self.net(x)

    def init_weights(self):
        def _wi(m):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        self.net.apply(_wi)
