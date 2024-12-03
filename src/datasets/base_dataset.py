import logging
import random
from typing import List

import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset

from src.model import MelSpectrogram, MelSpectrogramConfig, FastSpeechWrapper

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self, index, target_sr=16000, audio_length_limit=44100, rand_split=True, limit=None, shuffle_index=False, instance_transforms=None, text_as_input=False,
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self._assert_index_is_valid(index)
        self.rand_split = rand_split
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index: List[dict] = index

        self.instance_transforms = instance_transforms
        self.target_sr = target_sr
        self.audio_length_limit = audio_length_limit
        self.mel_transform = MelSpectrogram(MelSpectrogramConfig())

        self.text_as_input = text_as_input
        if text_as_input:
            self.txt2spec = FastSpeechWrapper()

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        instance_data = {}

        data_dict = self._index[ind]
        audio_path = data_dict.get("path")
        text_path = data_dict.get("text_path")
        if audio_path is None:
            audio = None
        else:
            audio = self.load_audio(audio_path)
            if self.audio_length_limit is not None:
                audio = self.random_subaudio(audio, self.audio_length_limit)
            instance_data.update({"wav": audio})

        if "text" in data_dict:
            text = data_dict["text"]
        else:
            text = None

        instance_data.update({
            "text": text,
            "audio_path": audio_path,
            "text_path": text_path,
        })

        if self.text_as_input:
            spectrogram = self.txt2spec(text)
        else:
            spectrogram = self.get_spectrogram(audio)

        instance_data.update({"mel_spectrogram": spectrogram})

        instance_data = self.preprocess_data(instance_data, special_keys=["get_spectrogram"])

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def load_object(self, path):
        """
        Load object from disk.

        Args:
            path (str): path to the object.
        Returns:
            data_object (Tensor):
        """
        data_object = torch.load(path)
        return data_object

    def random_subaudio(self, audio, length):
        _, len_wav = audio.shape
        if len_wav < length:
            return audio
        if self.rand_split:
            l_ed = np.random.randint(low=0, high=len_wav - length)
        else:
            l_ed = 0
        return audio[..., l_ed: l_ed+length]

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def get_spectrogram(self, audio):
        """
        Special instance transform with a special key to
        get spectrogram from audio.

        Args:
            audio (Tensor): original audio.
        Returns:
            spectrogram (Tensor): spectrogram for the audio.
        """
        return self.mel_transform(audio)


    def preprocess_data(self, instance_data, special_keys=["get_spectrogram"]):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
            single_key: optional[str]: if set modifies only this key
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is None:
            return instance_data

        for transform_name in self.instance_transforms.keys():
            if transform_name in special_keys:
                continue  # skip special key
            instance_data[transform_name] = self.instance_transforms[
                transform_name
            ](instance_data[transform_name])
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index: list,
    ) -> list:
        """
        Filter some of the elements from the dataset depending on
        some condition.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset that satisfied the condition. The dict has
                required metadata information, such as label and object path.
        """
        # Filter logic
        pass

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        pass

    @staticmethod
    def _sort_index(index):
        """
        Sort index via some rules.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting and after filtering.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): sorted list, containing dict for each element
                of the dataset. The dict has required metadata information,
                such as label and object path.
        """
        return sorted(index, key=lambda x: x["KEY_FOR_SORTING"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
