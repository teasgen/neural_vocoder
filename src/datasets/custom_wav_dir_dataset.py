from pathlib import Path

from src.datasets.base_dataset import BaseDataset


class CustomWavDirDataset(BaseDataset):
    def __init__(self, wav_dir, *args, **kwargs):
        data = []
        for path in Path(wav_dir).iterdir():
            entry = {}
            entry["path"] = path
            data.append(entry)
        super().__init__(data, *args, **kwargs)
