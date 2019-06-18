from abc import abstractmethod
from typing import Iterable, NoReturn, Union

import numpy as np
from torch import Tensor
from torch.nn import Module


class BaseModel(Module):
    """
    Base class for all models.
    """

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Union[NoReturn, Tensor]:
        """
        Forward pass logic abstract method.

        Parameters
        ----------
        *inputs : torch.Tensor
            Tensors representing the transformed input data.

        Returns
        -------
        torch.Tensor
            The model output tensor.

        Raises
        ------
        NotImplementedError
            If not implemented in child class.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """
        Prints model summary and the number of trainable parameters.
        """
        model_parameters: Iterable[Tensor] = filter(
            lambda p: p.requires_grad, self.parameters()
        )
        params: int = sum([np.prod(p.size()) for p in model_parameters])

        return super().__str__() + "\nTrainable parameters: {}".format(params)
