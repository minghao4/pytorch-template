from pathlib import Path
from typing import Union

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms.transforms import Compose

from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader.

    Attributes
    ----------
    data_dir : pathlib.Path or str
        The directory path to the data.

    dataset : torch.utils.data.Dataset
        The dataset object.
    """

    def __init__(
        self,
        data_dir: Union[Path, str],
        batch_size: int,
        shuffle: bool = True,
        validation_split: float = 0.0,
        num_workers: int = 1,
        training: bool = True,
    ):
        trsfm: Compose = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.data_dir: Union[Path, str] = data_dir
        self.dataset: Dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm
        )

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
