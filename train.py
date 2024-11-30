import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    D_loss_function = instantiate(config.D_loss_function).to(device)
    G_loss_function = instantiate(config.G_loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizers, learning rate schedulers
    discriminator_trainable_params = (
        list(filter(lambda p: p.requires_grad, model.mpd_discriminator.parameters())) + \
        list(filter(lambda p: p.requires_grad, model.msd_discriminator.parameters()))
    )
    
    D_optimizer = instantiate(config.D_optimizer, params=discriminator_trainable_params)
    D_lr_scheduler = instantiate(config.D_lr_scheduler, optimizer=D_optimizer)
    
    generator_trainable_params = filter(lambda p: p.requires_grad, model.generator.parameters())
    G_optimizer = instantiate(config.G_optimizer, params=generator_trainable_params)
    G_lr_scheduler = instantiate(config.G_lr_scheduler, optimizer=G_optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        D_criterion=D_loss_function,
        G_criterion=G_loss_function,
        metrics=metrics,
        D_optimizer=D_optimizer,
        G_optimizer=G_optimizer,
        D_lr_scheduler=D_lr_scheduler,
        G_lr_scheduler=G_lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
