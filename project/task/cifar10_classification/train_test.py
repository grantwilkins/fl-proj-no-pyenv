# noqa:EXE002
"""CIFAR10 training and testing functions, local and federated."""

from collections.abc import Sized
from pathlib import Path
from typing import cast
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD

from pydantic import BaseModel

from project.task.default.train_test import get_fed_eval_fn as get_default_fed_eval_fn
from project.task.default.train_test import (
    get_on_evaluate_config_fn as get_default_on_evaluate_config_fn,
)
from project.task.default.train_test import (
    get_on_fit_config_fn as get_default_on_fit_config_fn,
)
from project.types.common import IsolatedRNG

from tqdm import tqdm

from gauss_newton import DGN, BDGN
from cg_newton import CGN

from backpack import extend, backpack
from backpack.extensions import DiagGGNExact, DiagGGNMC, KFLR, KFAC, GGNMP


class TrainConfig(BaseModel):
    """Training configuration, allows '.' member access and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device
    epochs: int
    learning_rate: float
    optimizer: str

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def train(  # pylint: disable=too-many-arguments
    net: nn.Module,
    trainloader: DataLoader,
    _config: dict,
    _working_dir: Path,
    _rng_tuple: IsolatedRNG,
) -> tuple[int, dict]:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    _config : Dict
        The configuration for the training.
        Contains the device, number of epochs and learning rate.
        Static type checking is done by the TrainConfig class.
    _working_dir : Path
        The working directory for the training.
        Unused.
    _rng_tuple : IsolatedRNGTuple
        The random number generator state for the training.
        Use if you need seeded random behavior

    Returns
    -------
    Tuple[int, Dict]
        The number of samples used for training,
        the loss, and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, trainloader.dataset)) == 0:
        raise ValueError(
            "Trainloader can't be 0, exiting...",
        )

    config: TrainConfig = TrainConfig(**_config)
    del _config

    method = config.optimizer

    if method in {"diag_exact", "diag_mc", "block_exact", "block_mc"}:
        step_size = 0.05
        damping = 1.0
        if method == "diag_exact":
            opt = partial(DGN, step_size=step_size, damping=damping, mc=False)
            bp_extension = DiagGGNExact()
        elif method == "diag_mc":
            opt = partial(DGN, step_size=step_size, damping=damping, mc=True)
            bp_extension = DiagGGNMC()
        elif method == "block_exact":
            opt = partial(BDGN, step_size=step_size, damping=damping, mc=False)
            bp_extension = KFLR()
        elif method == "block_mc":
            opt = partial(BDGN, step_size=step_size, damping=damping, mc=True)
            bp_extension = KFAC()
    elif method == "sgd":
        opt = partial(
            SGD,
            lr=config.learning_rate,
            weight_decay=0.001,
        )
        bp_extension = None
    elif method == "adam":
        opt = partial(
            torch.optim.Adam,
            lr=config.learning_rate,
            weight_decay=0.001,
        )
        bp_extension = None
    elif method == "lbfgs":
        opt = partial(
            torch.optim.LBFGS,
            lr=config.learning_rate,
            max_iter=20,
            history_size=100,
            line_search_fn="strong_wolfe",
        )
        bp_extension = None
    elif method == "cg":
        bp_extension = GGNMP()
        opt = partial(
            CGN,
            bp_extension,
            lr=config.learning_rate,
            damping=1e-2,
            maxiter=0.1,
            tol=1e-6,
            atol=20,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    criterion = nn.CrossEntropyLoss().to(config.device)
    extend(criterion)

    net.to(config.device)
    net.eval()
    net = extend(net, use_converter=True)
    optimizer = opt(net.parameters())

    final_epoch_per_sample_loss = 0.0
    num_correct = 0
    for _ in range(config.epochs):
        final_epoch_per_sample_loss = 0.0
        num_correct = 0
        for batch_data, batch_target in tqdm(trainloader):
            data, target = (
                batch_data.to(config.device),
                batch_target.to(config.device),
            )
            if isinstance(optimizer, torch.optim.LBFGS):
                closure_final_epoch_per_sample_loss = 0.0
                closure_num_correct = 0

                def lbfgs_closure(
                    data: torch.Tensor = data, target: torch.Tensor = target
                ) -> torch.Tensor:
                    nonlocal closure_final_epoch_per_sample_loss, closure_num_correct
                    optimizer.zero_grad()
                    output = net(data)
                    loss = criterion(output, target)
                    closure_final_epoch_per_sample_loss += loss.item()
                    closure_num_correct += (
                        (output.max(1)[1] == target).clone().detach().sum().item()
                    )
                    loss.backward()
                    return loss

                optimizer.step(lbfgs_closure)
                final_epoch_per_sample_loss += closure_final_epoch_per_sample_loss
                num_correct += closure_num_correct
            else:
                optimizer.zero_grad()
                output = net(data)
                loss = criterion(output, target)
                final_epoch_per_sample_loss += loss.item()
                num_correct += (
                    (output.max(1)[1] == target).clone().detach().sum().item()
                )
                if bp_extension:
                    with backpack(bp_extension):
                        loss.backward()
                else:
                    loss.backward()
                optimizer.step()

    return len(cast(Sized, trainloader.dataset)), {
        "train_loss": final_epoch_per_sample_loss
        / len(cast(Sized, trainloader.dataset)),
        "train_accuracy": float(num_correct) / len(cast(Sized, trainloader.dataset)),
    }


class TestConfig(BaseModel):
    """Testing configuration, allows '.' member access and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def test(
    net: nn.Module,
    testloader: DataLoader,
    _config: dict,
    _working_dir: Path,
    _rng_tuple: IsolatedRNG,
) -> tuple[float, int, dict]:
    """Evaluate the network on the test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    _config : Dict
        The configuration for the testing.
        Contains the device.
        Static type checking is done by the TestConfig class.
    _working_dir : Path
        The working directory for the training.
        Unused.
    _rng_tuple : IsolatedRNGTuple
        The random number generator state for the training.
        Use if you need seeded random behavior


    Returns
    -------
    Tuple[float, int, float]
        The loss, number of test samples,
        and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, testloader.dataset)) == 0:
        raise ValueError(
            "Testloader can't be 0, exiting...",
        )

    config: TestConfig = TestConfig(**_config)
    del _config

    net.to(config.device)
    net.eval()

    criterion = nn.CrossEntropyLoss()
    correct, per_sample_loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = (
                images.to(
                    config.device,
                ),
                labels.to(config.device),
            )
            outputs = net(images)
            per_sample_loss += criterion(
                outputs,
                labels,
            ).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    return (
        per_sample_loss / len(cast(Sized, testloader.dataset)),
        len(cast(Sized, testloader.dataset)),
        {
            "test_accuracy": float(correct) / len(cast(Sized, testloader.dataset)),
        },
    )


# Use defaults as they are completely determined
# by the other functions defined in cifar10_classification
get_fed_eval_fn = get_default_fed_eval_fn
get_on_fit_config_fn = get_default_on_fit_config_fn
get_on_evaluate_config_fn = get_default_on_evaluate_config_fn
