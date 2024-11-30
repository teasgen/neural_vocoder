import gc

import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.D_optimizer.zero_grad()

        self.model.detach_generator()
        outputs = self.model(**batch)
        batch.update(outputs)
        del outputs

        all_losses = self.D_criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss_D"].backward()
            self._clip_grad_norm()
            self.D_optimizer.step()
            if self.D_lr_scheduler is not None:
                self.D_lr_scheduler.step()
            
            self.G_optimizer.zero_grad()

        self.model.train_generator()
        outputs = self.model(**batch)
        batch.update(outputs)
        del outputs

        all_losses = self.G_criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss_G"].backward()
            self._clip_grad_norm()
            self.G_optimizer.step()
            if self.G_lr_scheduler is not None:
                self.G_lr_scheduler.step()
        # update metrics for each loss
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        torch.cuda.empty_cache()
        gc.collect()
        # print(batch.keys())
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
