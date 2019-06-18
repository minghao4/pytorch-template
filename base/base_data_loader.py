from typing import Any, Callable, Dict, NoReturn, Optional, Tuple, Union

import numpy as np
from numpy import ndarray
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders

    Attributes
    ----------
    dataset : torch.utils.data.Dataset
        Training data.

    batch_size : int
        Number of samples per batch.

    shuffle : bool
        Whether or not to shuffle the loaded data.

    validation_split : float or int
        The split proportion or the exact number of samples of the validation set.

    num_workers : int
        Number of subprocesses to use for data loading.

    collate_fn: typing.Callable
        A function to merge a list of samples to form a mini-batch.

    Methods
    -------
    split_validation()
        Loads the validation data if the data is split into training and validation sets.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        validation_split: Union[float, int],
        num_workers: int,
        collate_fn: Callable = default_collate,
    ) -> None:

        self.validation_split: Union[float, int] = validation_split
        self.shuffle: bool = shuffle

        self.batch_idx: int = 0
        self.n_samples: int = len(dataset)

        # Setting the training/validation set samplers. Both samplers are `None` if validation split
        # is set to 0.
        self.sampler: Optional[SubsetRandomSampler]
        self.valid_sampler: Optional[SubsetRandomSampler]
        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        # Keyword arguments for the torch `DataLoader`.
        self.init_kwargs: Dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
        }

        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(
        self, split: Union[float, int]
    ) -> Union[NoReturn, Tuple[Optional[SubsetRandomSampler], Optional[SubsetRandomSampler]]]:
        """
        Randomly splits data indices into training and validation sets and create the
        corresponding sampler objects. Throws error if the validation set size is configured to be
        larger than the entire dataset.

        Parameters
        ----------
        split : float or int
            The split proportion or the exact number of samples of the validation set.

        Returns
        -------
        tuple
            The training and validation `SubsetRandomSampler` objects, in that order. A tuple of
            `None` if split value is 0.

        Raises
        ------
        AssertionError
            If validation set size is larger than entire dataset.
        """
        if split == 0.0:
            return None, None

        # Array of all sample indices.
        idx_full: ndarray = np.arange(self.n_samples)
        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert (
                split < self.n_samples
            ), "Validation set size is configured to be larger than entire dataset."
            len_valid: int = split  # number of validation samples
        else:
            len_valid = int(self.n_samples * split)

        # Sample indices for the training and validation sets.
        valid_idx: ndarray = idx_full[0:len_valid]
        train_idx: ndarray = np.delete(idx_full, np.arange(0, len_valid))

        # Training and validation set sampler objects.
        train_sampler: SubsetRandomSampler = SubsetRandomSampler(train_idx)
        valid_sampler: SubsetRandomSampler = SubsetRandomSampler(valid_idx)

        # Turn off shuffle option which is mutually exclusive with sampler option.
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self) -> Optional[DataLoader]:
        """
        Loads the validation data if the data is split.

        Returns
        -------
        torch.utils.data.DataLoader
            A torch `DataLoader` object with the validation `SubsetRandomSampler` object and
            keyword arguments.
        """
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
