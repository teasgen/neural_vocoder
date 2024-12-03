from pathlib import Path

from src.datasets.base_dataset import BaseDataset


class CustomTextDirDataset(BaseDataset):
    def __init__(self, transcription_dir, *args, **kwargs):
        data = []
        for path in Path(transcription_dir).iterdir():
            entry = {}
            with path.open() as f:
                entry["text"] = f.read().strip()
            entry["text_path"] = path
            data.append(entry)
        super().__init__(data, *args, **kwargs)
