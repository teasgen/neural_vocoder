import torch
import torch.nn as nn

from .generator import Generator
from .discriminators import MPD, MSD
from .mel_spec import MelSpectrogram, MelSpectrogramConfig


class HiFiGAN(nn.Module):
    def __init__(self, mel_channels: int = 80, hu: int = 128, ku: list[int] = [16, 16, 4, 4], kr: list[int] = [3, 7, 11], dr: list[int] = [1, 3, 5]):
        super().__init__()

        self.generator = Generator(
            mel_channels=mel_channels,
            hu=hu,
            ku=ku,
            kr=kr,
            dr=dr,
        )

        # periods from https://arxiv.org/pdf/2010.05646
        self.mpd_discriminator = MPD(periods=[2, 3, 5, 7, 11])
        self.msd_discriminator = MSD()

        self.mel_transform = MelSpectrogram(MelSpectrogramConfig())
        self._freeze_gen = False

    def freeze_gen(self, freeze=True):
        self._freeze_gen = freeze

    def forward(self, mel_spectrogram, **batch):
        gen_wav = self.generator(mel_spectrogram)
        if self._freeze_gen:
            gen_wav = gen_wav.detach()

        wav = batch.get("wav")

        # inference
        if wav is None:
            return {
                "gen_wav": gen_wav
            }
        if len(wav.shape) == 2:
            wav = wav.unsqueeze(1)
        if len(mel_spectrogram.shape) == 3:
            mel_spectrogram = mel_spectrogram.unsqueeze(1)

        gen_wav = gen_wav[..., : wav.shape[-1]]

        gen_spec = self.mel_transform(gen_wav)

        msd_gen_features, msd_gen_map_features = self.msd_discriminator(gen_wav)
        mpd_gen_features, mpd_gen_map_features = self.mpd_discriminator(gen_wav)

        msd_gt_features, msd_gt_map_features = self.msd_discriminator(wav)
        mpd_gt_features, mpd_gt_map_features = self.mpd_discriminator(wav)

        return {
            "wav": wav,
            "spec": mel_spectrogram,
            
            "msd_gt_features": msd_gt_features,
            "msd_gt_map_features": msd_gt_map_features,
            
            "mpd_gt_features": mpd_gt_features,
            "mpd_gt_map_features": mpd_gt_map_features,
            
            "gen_wav": gen_wav,
            "gen_spec": gen_spec,
            
            "msd_gen_features": msd_gen_features,
            "msd_gen_map_features": msd_gen_map_features,
            
            "mpd_gen_features": mpd_gen_features,
            "mpd_gen_map_features": mpd_gen_map_features,
        }
    
    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info