import torch
from torch import nn
import torch.nn.functional as F


class HiFiGANLossGenerator(nn.Module):
    def __init__(self, lambda_fm = 2, lambda_mel = 45):
        super().__init__()
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel


    def feature_matching_loss(self, gt, pred):
        """
        gt (pred) - List of Discriminators - List of layers
        """
        loss = 0.0
        for disc_gt, disc_pred in zip(gt, pred):
            for disc_layer_gt, disc_layer_pred in zip(disc_gt, disc_pred):
                loss += F.l1_loss(disc_layer_gt, disc_layer_pred)
        return loss
    

    def generator_loss(self, disc_pred_out):
        loss = 0.0
        for disc_pred in disc_pred_out:
            loss += torch.mean((disc_pred - 1) ** 2)
        return loss


    def forward(
        self,
        spec,
        msd_gt_map_features,
        mpd_gt_map_features,
        gen_spec,
        msd_gen_features,
        msd_gen_map_features,
        mpd_gen_features,
        mpd_gen_map_features,
        **batch
    ):
        loss_mel = F.l1_loss(spec, gen_spec)

        loss_gen = self.generator_loss(msd_gen_features) + self.generator_loss(mpd_gen_features)

        loss_feature_matching = self.feature_matching_loss(msd_gt_map_features, msd_gen_map_features) + \
            self.feature_matching_loss(mpd_gt_map_features, mpd_gen_map_features)

        loss_G = loss_gen + self.lambda_fm * loss_feature_matching + self.lambda_mel * loss_mel

        return {
            "loss_G": loss_G,
        }

class HiFiGANLossDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def discriminator_loss(self, disc_gt_out, disc_pred_out):
        loss = 0.0
        for disc_gt, disc_pred in zip(disc_gt_out, disc_pred_out):
            loss += (torch.mean((disc_gt - 1) ** 2) + torch.mean(disc_pred ** 2))
        return loss


    def forward(
        self,
        msd_gt_features,
        mpd_gt_features,
        msd_gen_features,
        mpd_gen_features,
        **batch
    ):
        loss_disc = self.discriminator_loss(msd_gt_features, msd_gen_features) + self.discriminator_loss(mpd_gt_features, mpd_gen_features)

        loss_D = loss_disc

        return {
            "loss_D": loss_D,
        }
