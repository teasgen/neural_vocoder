from pathlib import Path

from src.datasets.base_dataset import BaseDataset


class CustomSingleTextDataset(BaseDataset):
    def __init__(self, transcription, *args, **kwargs):
        data = []
        entry = {}
        entry["text"] = transcription
        entry["text_path"] = "custom_text.txt"
        data.append(entry)
        super().__init__(data, *args, **kwargs)
