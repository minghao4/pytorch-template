from torch import Tensor
import torch.nn.functional as F


# This file is for adapting loss functions as callables that can be reached with a single string.
def nll_loss(output: Tensor, target: Tensor) -> Tensor:
    """Negative log-likelihood loss calculation."""
    return F.nll_loss(output, target)
