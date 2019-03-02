# losely based on Temporal Convolutional Networks (TCN)

import torch

from torchvision import models
from torchvision import transforms
import torch.nn.functional as F
from torch import autograd
import numpy as np

from typing import *
from torch import nn

from PIL import Image

from functools import partial

import kraken.lib.ctc_decoder

class TorchSeqRecognizer(object):
    """
    A class wrapping a TorchVGSLModel with a more comfortable recognition interface.
    """
    def __init__(self, nn, codec=None, decoder=kraken.lib.ctc_decoder.greedy_decoder, train: bool = False, device: str = 'cpu') -> None:
        """
        Constructs a sequence recognizer from a VGSL model and a decoder.

        Args:
            nn (kraken.lib.vgsl.TorchVGSLModel): neural network used for recognition
            decoder (func): Decoder function used for mapping softmax
                            activations to labels and positions
            train (bool): Enables or disables gradient calculation
            device (torch.Device): Device to run model on
        """
        self.nn = nn
        self.kind = ''
        if train:
            self.nn.train()
        else:
            self.nn.eval()
        self.codec = codec
        self.decoder = decoder
        self.train = train
        self.device = device
        self.nn.to(device)

    def to(self, device):
        """
        Moves model to device and automatically loads input tensors onto it.
        """
        self.device = device
        self.nn.to(device)

    def forward(self, line: torch.Tensor) -> np.array:
        """
        Performs a forward pass on a torch tensor of a line with shape (C, H, W)
        and returns a numpy array (W, C).
        """
        # make CHW -> 1CHW
        line = line.to(self.device)
        line = line.unsqueeze(0)
        o = self.nn(line)
        if o.size(2) != 1:
            raise KrakenInputException('Expected dimension 3 to be 1, actual {}'.format(o.size()))
        self.outputs = o.detach().squeeze().cpu().numpy()
        return self.outputs

    def predict(self, line: torch.Tensor) -> List[Tuple[str, int, int, float]]:
        """
        Performs a forward pass on a torch tensor of a line with shape (C, H, W)
        and returns the decoding as a list of tuples (string, start, end,
        confidence).
        """
        o = self.forward(line)
        locs = self.decoder(o)
        return self.codec.decode(locs)

    def predict_string(self, line: torch.Tensor) -> str:
        """
        Performs a forward pass on a torch tensor of a line with shape (C, H, W)
        and returns a string of the results.
        """
        o = self.forward(line)
        locs = self.decoder(o)
        decoding = self.codec.decode(locs)
        return ''.join(x[0] for x in decoding)

    def predict_labels(self, line: torch.tensor) -> List[Tuple[int, int, int, float]]:
        """
        Performs a forward pass on a torch tensor of a line with shape (C, H, W)
        and returns a list of tuples (class, start, end, max). Max is the
        maximum value of the softmax layer in the region.
        """
        o = self.forward(line)
        return self.decoder(o)


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
        padding = tuple(d*(k - 1) // 2 for k, d in zip(kernel_size, dilation))
        self.layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation),
                                   nn.ReLU(),
                                   reg_l(),
                                   nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation),
                                   nn.ReLU(),
                                   reg_l(),
                                   nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation),
                                   reg_l())
        # downsampling for residual
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        o = self.layer(x)
        o = torch.relu(o + self.residual(x) if self.residual else o + x)
        return o

class TDNNBlock(nn.Module):

    def __init__(self, in_channels, in_features, out_features, kernel_size, dilation):
        super(TDNNBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, dilation=(1, dilation), padding=(0, 2*(dilation-1))), nn.Dropout(0.1))
        self.linear = nn.Sequential(nn.Linear(in_features, out_features), nn.Dropout(0.5))

    def forward(self, x):
        o = self.conv(x)
        # NCHW -> NW(C*H)
        o = o.reshape(o.shape[0], -1, o.shape[3]).transpose(1, 2)
        o = F.relu(self.linear(o))
        return o.unsqueeze(2).transpose(1, 3)

class FinalBlock(nn.Module):
    def __init__(self):
        super(FinalBlock, self).__init__()

    def forward(self, x):
        if not self.training:
            o = F.softmax(x, dim=3)
        else:
            o = F.log_softmax(x, dim=3)
        return o

class CausalNet(nn.Module):

    def __init__(self, input_size, output_size, out_channels=(64, 128, 256), layers=3, kernel_sizes=((7, 7), (5, 5), (3, 3)), dropout=0.1, reg='dropout'):
        super(CausalNet, self).__init__()
        l = []
        l.append(CausalBlock(1, out_channels[0], kernel_sizes[0], stride=1, dilation=(1, 1), dropout=dropout))
        for i in range(layers-1):
            l.append(nn.AvgPool2d(2))
            dilation_size = 2 ** (i+1)
            l.append(CausalBlock(out_channels[i], out_channels[i+1], kernel_sizes[i+1],
                                 stride=1, dilation=(1, dilation_size),
                                 dropout=dropout, reg=reg))
        self.encoder = nn.Sequential(*l)
        self.decoder = nn.Linear(input_size//(layers+1) * out_channels[-1], output_size)
        self.init_weights()

    def forward(self, x):
        o = self.encoder(x)
        return self.decoder(o.reshape(o.shape[0], -1, o.shape[3]).transpose(1, 2))

    def init_weights(self):
        def _wi(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        self.encoder.apply(_wi)
        self.decoder.apply(_wi)
