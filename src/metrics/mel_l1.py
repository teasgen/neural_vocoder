import torch

from src.metrics.base_metric import BaseMetric


class L1MelMetric(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = torch.nn.functional.l1_loss

    def __call__(self, spec: torch.Tensor, gen_spec: torch.Tensor, **kwargs):
        return self.metric(spec, gen_spec)
