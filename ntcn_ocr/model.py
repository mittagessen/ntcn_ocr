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
class MultiParamSequential(nn.Sequential):
    """
    Sequential variant accepting multiple arguments.
    """
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class DilConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1)//2
        self.stride = stride
        self.block_b = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size, stride=self.stride, padding=self.padding),
                                     nn.GroupNorm(32, out_channels),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
                                     nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=self.padding),
                                     nn.GroupNorm(32, out_channels))
        self.block_a = nn.Sequential(nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=self.padding),
                                     nn.GroupNorm(32, out_channels),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
                                     nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=self.padding),
                                     nn.GroupNorm(32, out_channels))

        # downsampling for residual
        self.residual = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride), nn.GroupNorm(32, out_channels))

    def forward(self, x, seq_lens):
        o = self.block_b(x)
        o = torch.relu(o + self.residual(x) if self.residual else o + x)
        o_b = self.block_a(o)
        o_b = torch.relu(o_b + o)
        seq_lens = torch.clamp(torch.floor((seq_lens+2*self.padding-(self.kernel_size-1)-1).float()/self.stride+1), min=1).int()
        return o_b, seq_lens


class ConvSeqNet(nn.Module):

    def __init__(self, input_size, output_size, out_channels=(64, 128, 256), kernel_sizes=(3, 3, 3), blocks=3):
        super().__init__()
        l = []
        l.append(DilConvBlock(input_size, out_channels[0], kernel_sizes[0], stride=1))
        for i in range(blocks-1):
            l.append(DilConvBlock(out_channels[i], out_channels[i+1], kernel_sizes[i+1], stride=2))
        self.encoder = MultiParamSequential(*l)
        self.decoder = nn.Conv1d(out_channels[-1], output_size, 1)
        self.init_weights()

    def forward(self, x, seq_lens):
        o, seq_lens = self.encoder(x, seq_lens)
        o = self.decoder(o)
        if not self.training:
            o = F.softmax(o, dim=2)
        else:
            o = F.log_softmax(o, dim=2)
        return o, seq_lens

    def init_weights(self):
        def _wi(m):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        self.encoder.apply(_wi)
        self.decoder.apply(_wi)

class TorchSeqRecognizer(object):
    """
    A class wrapping a TorchVGSLModel with a more comfortable recognition interface.
    """
    def __init__(self, nn, codec, decoder=kraken.lib.ctc_decoder.greedy_decoder, train: bool = False, device: str = 'cpu') -> None:
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

    def forward(self, line: torch.Tensor, lens: torch.Tensor = None) -> np.array:
        """
        Performs a forward pass on a torch tensor of one or more lines with
        shape (N, C, H, W) and returns a numpy array (N, W, C).

        Args:
            line (torch.Tensor): NCHW line tensor
            lens (torch.Tensor): Optional tensor containing sequence lengths if N > 1

        Returns:
            Tuple with (N, W, C) shaped numpy array and final output sequence
            lengths.
        """
        line = line.to(self.device)
        o, olens = self.nn(line, lens)
        self.outputs = o.detach().cpu().numpy()
        if olens is not None:
            olens = olens.cpu().numpy()
        return self.outputs, olens

    def predict(self, line: torch.Tensor, lens: torch.Tensor = None) -> List[List[Tuple[str, int, int, float]]]:
        """
        Performs a forward pass on a torch tensor of a line with shape (N, C, H, W)
        and returns the decoding as a list of tuples (string, start, end,
        confidence).

        Args:
            line (torch.Tensor): NCHW line tensor
            lens (torch.Tensor): Optional tensor containing sequence lengths if N > 1

        Returns:
            List of decoded sequences.
        """
        o, olens = self.forward(line, lens)
        dec_seqs = []
        if olens is not None:
            for seq, seq_len in zip(o, olens):
                locs = self.decoder(seq[:, :seq_len])
                dec_seqs.append(self.codec.decode(locs))
        else:
            locs = self.decoder(o[0])
            dec_seqs.append(self.codec.decode(locs))
        return dec_seqs

    def predict_string(self, line: torch.Tensor, lens: torch.Tensor = None) -> List[str]:
        """
        Performs a forward pass on a torch tensor of a line with shape (N, C, H, W)
        and returns a string of the results.
        """
        o, olens = self.forward(line, lens)
        dec_strs = []
        if olens is not None:
            for seq, seq_len in zip(o, olens):
                locs = self.decoder(seq[:, :seq_len])
                dec_strs.append(''.join(x[0] for x in self.codec.decode(locs)))
        else:
            locs = self.decoder(o[0])
            dec_strs.append(''.join(x[0] for x in self.codec.decode(locs)))
        return dec_strs

    def predict_labels(self, line: torch.tensor, lens: torch.Tensor = None) -> List[List[Tuple[int, int, int, float]]]:
        """
        Performs a forward pass on a torch tensor of a line with shape (N, C, H, W)
        and returns a list of tuples (class, start, end, max). Max is the
        maximum value of the softmax layer in the region.
        """
        o, olens = self.forward(line, lens)
        oseqs = []
        if olens is not None:
            for seq, seq_len in zip(o, olens):
                oseqs.append(self.decoder(seq[:, :seq_len]))
        else:
            oseqs.append(self.decoder(o[0]))
        return oseqs
