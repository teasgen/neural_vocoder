import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dr):
        super().__init__()
        
        net = []
        for dilation in dr:
            net += [
                nn.LeakyReLU(),
                nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding="same"),
                nn.LeakyReLU(),
                nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding="same"),
            ]
        self.model = nn.Sequential(*net)

    def forward(self, x):
        """
        Args:
            x (bs, channels, time)
        """
        return self.model(x) + x


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
            out += self.models[idx](x)

        return out



class Generator(nn.Module):
    """
    Default parameters from V2 HiFiGAN version
    """
    def __init__(self, mel_channels: int = 80, hu: int = 128, ku: list[int] = [16, 16, 4, 4], kr: list[int] = [3, 7, 11], dr: list[int] = [1, 3, 5]):
        super().__init__()

        self.input_conv = nn.Conv1d(mel_channels, hu, 7, padding="same")

        self.generator = nn.Sequential()

        current_channels = hu
        
        for transpose_kernel_size in ku:
            stride = transpose_kernel_size // 2
            channels = current_channels // 2
            transposed_conv = nn.ConvTranspose1d(
                in_channels=current_channels,
                out_channels=channels,
                kernel_size=transpose_kernel_size,
                stride=stride,
                padding=(transpose_kernel_size - stride) // 2,
            )
            mrf = MRF(channels=channels, kr=kr, dr=dr)
            self.generator.append(nn.Sequential(
                nn.LeakyReLU(),
                transposed_conv,
                mrf,
            ))

            current_channels = channels


        self.out_conv = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(current_channels, 1, 7, padding="same"),
            nn.Tanh(),
        )

    def forward(self, mel_spectrogram):
        """
        mel_spectrogram (b, c, t_mel) -> audio (b, 1, t_audio)
        """
        x = self.input_conv(mel_spectrogram)
        x = self.generator(x)
        x = self.out_conv(x)
        return x
