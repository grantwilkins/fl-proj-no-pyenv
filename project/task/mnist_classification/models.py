"""CNN model architecture, training, and testing functions for MNIST."""

import torch
from torch import nn
from project.types.common import NetGen
from project.utils.utils import lazy_config_wrapper

Net = nn.Sequential(
    nn.Conv2d(1, 32, 5, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), padding=1),
    nn.Conv2d(32, 64, 5, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), padding=1),
    nn.Flatten(1),
    nn.Linear(64 * 7 * 7, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)

get_net: NetGen = lambda x, y: Net  # noqa: E731,ARG005


class LogisticRegression(nn.Module):
    """A network for logistic regression using a single fully connected layer.

    As described in the Li et al., 2020 paper :

    [Federated Optimization in Heterogeneous Networks] (

    https://arxiv.org/pdf/1812.06127.pdf)
    """

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize the network.

        Parameters
        ----------
        num_classes : int
            Number of classes in the dataset.

        Returns
        -------
        None
        """
        super().__init__()
        self.linear = nn.Linear(28 * 28, num_classes)

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        output_tensor = self.linear(
            torch.flatten(input_tensor, 1),
        )
        return output_tensor


# Simple wrapper to match the NetGenerator Interface
get_logistic_regression: NetGen = lazy_config_wrapper(
    LogisticRegression,
)
