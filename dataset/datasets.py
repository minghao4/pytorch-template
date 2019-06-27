import os
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class ExampleDataset(Dataset):
    """
    Lipid dataset example class.

    Attributes
    ----------
    data_len : int
        Sample size.

    feats : int
        Number of features.

    genotypes : torch.Tensor
        Genotype data.

    phenotypes : torch.Tensor
        Phenotype data.
    """

    def __init__(self, root: str, train: bool, transforms=None) -> None:
        # File paths.
        phenotype_dir: str = os.path.basename(root)
        train_test: str = "test"
        if train:
            train_test = "train"

        input_file: str = os.path.join(
            root, train_test,
            "lipids_genotype_" + phenotype_dir + '_' + train_test + ".pth"
        )
        target_file: str = os.path.join(
            root, train_test,
            "lipids_phenotype_" + phenotype_dir + '_' + train_test + ".pth"
        )

        # Data loading.
        self.genotypes: torch.Tensor = torch.load(input_file)[:, 1:]
        self.phenotypes: torch.Tensor = torch.load(target_file)[:, 1].unsqueeze(1)
        print("Data loading complete.")

        self.data_len: int = self.genotypes.shape[0]  # sample size
        self.feats: int = self.genotypes.shape[1]  # number of encoded features

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X: torch.Tensor = self.genotypes[idx]
        y: torch.Tensor = self.phenotypes[idx]
        return (X, y)

    def __len__(self) -> int:
        return self.data_len
