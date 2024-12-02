import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .utils import init_norm_weights


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dr):
        super().__init__()

        net = []
        for dilation in dr:
            net += [
                weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding="same")),
                weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=1, padding="same")),
            ]
        self.act = nn.LeakyReLU(0.1)
        self.model = nn.Sequential(*net)
        self.model.apply(init_norm_weights)

    def forward(self, x):
        """
        Args:
            x (bs, channels, time)
        """
        for idx in range(0, len(self.model), 2):
            residual = x
            x = self.model[idx](self.act(x))
            x = self.model[idx + 1](self.act(x))
            x = x + residual
        return x


class MRF(nn.Module):
    def __init__(self, channels, kr, dr):
        super().__init__()

        net = []
        for kernel_size in kr:
            net += [
                ResBlock(channels, kernel_size, dr),
            ]
        self.models = nn.ModuleList(net)

    def forward(self, x):
        """
        Args:
            x (bs, channels, time)
        """
        out = self.models[0](x)
        for idx in range(1, len(self.models)):
            out = out + self.models[idx](x)

        return out / len(self.models)



class Generator(nn.Module):
    """
    Default parameters from V2 HiFiGAN version
    """
    def __init__(self, mel_channels: int = 80, hu: int = 128, ku: list[int] = [16, 16, 4, 4], kr: list[int] = [3, 7, 11], dr: list[int] = [1, 3, 5]):
        super().__init__()

        self.input_conv = weight_norm(nn.Conv1d(mel_channels, hu, 7, padding="same"))

        self.generator = nn.Sequential()

        current_channels = hu
        
        for transpose_kernel_size in ku:
            stride = transpose_kernel_size // 2
            channels = current_channels // 2
            transposed_conv = weight_norm(nn.ConvTranspose1d(
                in_channels=current_channels,
                out_channels=channels,
                kernel_size=transpose_kernel_size,
                stride=stride,
                padding=(transpose_kernel_size - stride) // 2,
            ))
            mrf = MRF(channels=channels, kr=kr, dr=dr)
            self.generator.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                transposed_conv,
                mrf,
            ))

            current_channels = channels


        self.out_conv = nn.Sequential(
            nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(current_channels, 1, 7, padding="same")),
            nn.Tanh(),
        )

        self.generator.apply(init_norm_weights)
        self.out_conv.apply(init_norm_weights)

    def forward(self, mel_spectrogram):
        """
        mel_spectrogram (b, c, t_mel) -> audio (b, 1, t_audio)
        """
        x = self.input_conv(mel_spectrogram)
        x = self.generator(x)
        x = self.out_conv(x)
        return x