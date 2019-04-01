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
        self.outputs = o.detach().squeeze().cpu().numpy().T
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


class DilConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0.1, reg='dropout'):
        super().__init__()
        if reg == 'dropout2d':
            reg_l = partial(nn.Dropout2d, dropout)
        elif reg == 'dropout':
            reg_l = partial(nn.Dropout, dropout)
        elif reg == 'batchnorm':
            reg_l = partial(nn.BatchNorm2d, out_channels)
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

class TDNNLayer(nn.Module):
    """
    A TDNN layer with dropout and ReLU nonlinearity.

    Expects an input NHW and outputs NOW, where O is output_size. Taps are the
    offsets folded into the current time step with negative values
    corresponding to context to the left and positive values corresponding to
    context to the right.
    """
    def __init__(self, input_size, output_size, dropout=0.5, taps=(-4, 0, 4)):
        super().__init__()
        self.input_size = len(taps) * input_size
        self.output_size = output_size
        self.taps = taps
        self.lin = nn.Sequential(nn.Linear(self.input_size, self.output_size),
                                 nn.ReLU(),
                                 nn.Dropout(dropout))

    def forward(self, x):
        siz = x.shape
        # append the taps by padding to the left/right
        xs = []
        for tap in self.taps:
            if tap <= 0:
                p = (abs(tap), max(self.taps) + tap)
                offset = 0
            elif tap > 0:
                p = (0, max(self.taps) + tap)
                offset = tap
            xs.append(F.pad(x[:, :, offset:], p))
        # stack and discard padding no longer needed 
        x = torch.cat(xs, dim=1)[:, :, :siz[2]].transpose(1, 2)
        return self.lin(x).transpose(1, 2)


class ConvSeqNet(nn.Module):

    def __init__(self, input_size, output_size, out_channels=(36, 70, 70), layers=3, tdnn_hidden=500, kernel_sizes=((7, 7), (5, 5), (3, 3)), dropout=0.1, reg='dropout'):
        super().__init__()
        l = []
        l.append(DilConvBlock(1, out_channels[0], kernel_sizes[0], stride=1, dilation=(1, 1), dropout=dropout))
        for i in range(layers-1):
            l.append(nn.AvgPool2d(2))
            dilation_size = 2 ** (i+1)
            l.append(DilConvBlock(out_channels[i], out_channels[i+1], kernel_sizes[i+1],
                                 stride=1, dilation=(1, dilation_size),
                                 dropout=dropout, reg=reg))
        self.encoder = nn.Sequential(*l)
        self.decoder = nn.Sequential(TDNNLayer(input_size//(layers+1) * out_channels[-1], tdnn_hidden),
                                     TDNNLayer(tdnn_hidden, tdnn_hidden),
                                     TDNNLayer(tdnn_hidden, output_size))
        self.init_weights()

    def forward(self, x):
        o = self.encoder(x)
        o = self.decoder(o.reshape(o.shape[0], -1, o.shape[3]))
        if not self.training:
            o = F.softmax(o, dim=2)
        else:
            o = F.log_softmax(o, dim=2)
        return o

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
