import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


class SingleMPD(nn.Module):
    def __init__(self, period: int, norm_function: callable):
        super().__init__()
        self.period = period
        self.norm_function = norm_function

        self.convs = []
        last_ch = 1
        for idx in range(1, 5):
            now_ch = 2 ** (6 + idx)
            self.convs += self.single_block(last_ch, now_ch)
            last_ch = now_ch
        self.convs += self.single_block(last_ch, 1024)
        self.convs = nn.ModuleList(self.convs)
        self.act = nn.LeakyReLU(0.1)
        self.out_conv = self.norm_function(nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=(3, 1)))


    def single_block(self, ch_in, ch_out):
        return [
            self.norm_function(nn.Conv2d(ch_in, ch_out, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
        ]

    def forward(self, x):
        b, c, t = x.shape  # c = 1

        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), "reflect")
            t = t + pad

        x = x.reshape(b, c, t // self.period, self.period)

        features = []
        for module in self.convs:
            x = self.act(module(x))
            features += [x]

        x = self.out_conv(x)
        features += [x]

        x = torch.flatten(x, 1, -1)

        return x, features


class MPD(nn.Module):
    def __init__(self, periods: list[int]):
        super().__init__()
        norm_function = weight_norm
        self.mpds = nn.ModuleList([SingleMPD(period, norm_function=norm_function) for period in periods])

    def forward(self, x):
        """
        x (bs, 1, t_audio)
        """
        features = []
        all_maps = []

        for mpd in self.mpds:
            cur_x, cur_features = mpd(x)
            features += [cur_x] 
            all_maps += [cur_features] 

        return features, all_maps

class SingleMSD(nn.Module):
    """
    Parameters from https://arxiv.org/pdf/1910.06711
    """
    def __init__(self, norm_function: callable):
        super().__init__()
        self.norm_function = norm_function

        self.convs = nn.ModuleList([
            self.single_block(1, 16, kernel_size=15, stride=1),
            self.single_block(16, 64, kernel_size=41, stride=4, groups=4),
            self.single_block(64, 256, kernel_size=41, stride=4, groups=16),
            self.single_block(256, 1024, kernel_size=41, stride=4, groups=64),
            self.single_block(1024, 1024, kernel_size=41, stride=4, groups=256),
            self.single_block(1024, 1024, kernel_size=5, stride=4),
        ])
        self.act = nn.LeakyReLU(0.1)
        self.out_conv = self.norm_function(nn.Conv1d(1024, 1, kernel_size=3, padding="same"))

    def single_block(self, in_ch, out_ch, kernel_size, stride, groups=1):
        return self.norm_function(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, groups=groups, padding=kernel_size // 2)
        )
    
    def forward(self, x):
        features = []
        for module in self.convs:
            x = self.act(module(x))
            features += [x]

        x = self.out_conv(x)
        features += [x]

        x = torch.flatten(x, 1, -1)

        return x, features

class MSD(nn.Module):
    def __init__(self):
        super().__init__()

        self.msds = nn.ModuleList([
            SingleMSD(spectral_norm if idx == 0 else weight_norm) for idx in range(3)
        ])
        self.poolings = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(8, 4, padding=4),
        ])
    
    def forward(self, x):
        """
        x (bs, 1, t_audio)
        """
        features = []
        all_maps = []

        for pooling, msd in zip(self.poolings, self.msds):
            cur_x, cur_features = msd(pooling(x))
            features += [cur_x] 
            all_maps += [cur_features] 

        return features, all_maps
