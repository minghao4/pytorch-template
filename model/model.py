from torch import Tensor
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F

from base import BaseModel


class MnistModel(BaseModel):
    """
    Example CNN model to learn MNIST data.

    Attributes
    ----------
    num_classes : int, optional
        Number of classes present (default is 10)

    Methods
    -------
    forward()
        Forward pass through the network.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1: Module = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2: Module = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop: Module = nn.Dropout2d()
        self.fc1: Module = nn.Linear(320, 50)
        self.fc2: Module = nn.Linear(50, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass logic.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The model output tensor.
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
