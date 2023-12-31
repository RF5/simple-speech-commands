"""Imported from https://github.com/prlz77/ResNeXt.pytorch/blob/master/models/model.py
and added support for the 1x32x32 mel spectrogram for the speech recognition.
Creates a ResNeXt Model as defined in:
Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016).
Aggregated residual transformations for deep neural networks.
arXiv preprint arXiv:1611.05431.
"""

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

from random import sample
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch import Tensor
import librosa
import numpy as np
import torch

__all__ = [ 'CifarResNeXt', 'ResNextWrapper', 'CLASSES' ]

CLASSES = 'zero, one, two, three, four, five, six, seven, eight, nine'.split(', ')


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)


class CifarResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, nlabels, cardinality=8, depth=29, base_width=64, widen_factor=4, in_channels=1):
        """ Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(CifarResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.classifier = nn.Linear(self.stages[3], nlabels)
        init.kaiming_normal_(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.base_width, self.widen_factor))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width,
                                                   self.widen_factor))
        return block

    def forward(self, x: Tensor, return_features=False):
        """ Get prediction classifications from mel-spectrograms `x` (bs, 1, n_mels, seq_len) """
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        if return_features:
            return self.classifier(x), x
        return self.classifier(x)

class ResNextWrapper(nn.Module):

    def __init__(self, classes=10, sample_rate=16000) -> None:
        super().__init__()
        self.model = CifarResNeXt(classes)
        self.sample_rate = sample_rate

    @torch.inference_mode()
    def wav2mel(self, x: Tensor, n_fft=2048, hop_length=512, n_mels=32) -> Tensor:
        """ `x` is wavs of shape (N, T) """
        N = x.shape[0]
        out = []
        in_device = x.device
        for i in range(N):
            stft = librosa.stft(x[i].cpu().numpy(), n_fft=n_fft, hop_length=hop_length)
            mel_basis = librosa.filters.mel(sr=self.sample_rate, n_fft=n_fft, n_mels=n_mels)
            s = np.dot(mel_basis, np.abs(stft)**2.0)
            mel_spectrogram = librosa.power_to_db(s, ref=np.max)
            mel = torch.from_numpy(mel_spectrogram).to(in_device)
            out.append(mel)
        out = torch.stack(out, dim=0)
        return out

    def forward_mel(self, mel: Tensor, return_features=False) -> Tensor:
        """ Input mel spectrograms `mel` (bs, n_mels, seq_len) """
        if return_features:
            out, feats = self.model(mel[:, None], return_features=return_features)
            return out, feats
        else:
            out = self.model(mel[:, None])
        return out

    def forward(self, x: Tensor, xlen: Tensor, return_features=False) -> Tensor:
        """ Input waveforms `x` (bs, T) """
        mel = self.wav2mel(x)
        return self.forward_mel(mel, return_features=return_features)
