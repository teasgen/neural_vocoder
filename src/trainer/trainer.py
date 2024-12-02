from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.logger.utils import plot_spectrogram


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

        self.model.freeze_gen(True)
        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.D_criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss_D"].backward()
            self._clip_grad_norm(generator=False)
            self.D_optimizer.step()
            if self.D_lr_scheduler is not None:
                self.D_lr_scheduler.step()
            
            self.G_optimizer.zero_grad()

        self.model.freeze_gen(False)
        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.G_criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss_G"].backward()
            self._clip_grad_norm(generator=True)
            self.G_optimizer.step()
            if self.G_lr_scheduler is not None:
                self.G_lr_scheduler.step()
        # update metrics for each loss
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch).item())

        return batch

    def log_spectrogram(self, batch_idx, mode, mel_spectrogram, **batch):
        spectrogram_for_plot = mel_spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image(f"{mode}_true_spectrogram", image)

        if "gen_spec" in batch:
            spectrogram_for_plot = batch["gen_spec"][0][0].detach().cpu()
            image = plot_spectrogram(spectrogram_for_plot)
            self.writer.add_image(f"{mode}_gen_spectrogram", image)

    def log_audio(self, batch_idx, mode, gen_wav, **batch):
        self.writer.add_audio(f"{mode}_gen_audio", gen_wav[0], sample_rate=22050)
        if "wav" in batch:
            self.writer.add_audio(f"{mode}_true_audio", batch["wav"][0], sample_rate=22050)

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
        if mode == "train":
            self.log_spectrogram(batch_idx, mode, **batch)
            self.log_audio(batch_idx, mode, **batch)
        else:
            self.log_spectrogram(batch_idx, mode, **batch)
            self.log_audio(batch_idx, mode, **batch)
