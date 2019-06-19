from argparse import ArgumentParser
import collections
from logging import Logger
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer


def main(config: ConfigParser) -> None:
    """
    Main training function.

    Parameters
    ----------
    config : parse_config.ConfigParser
        Parsed configuration JSON file.
    """
    logger: Logger = config.get_logger("train")

    # Setup data_loader instances.
    data_loader: DataLoader = config.initialize("data_loader", module_data)
    valid_data_loader: Optional[DataLoader] = data_loader.split_validation()

    # Build model architecture, then print to console.
    model: Module = config.initialize("arch", module_arch)
    logger.info(model)

    # Get function handles of loss and metrics as well as args.
    loss_fn: Callable = getattr(module_loss, config["loss"]["type"])
    loss_args: Dict[str, Any] = config["loss"]["args"]
    metric_fns: List[Callable] = [getattr(module_metric, met) for met in config["metrics"]]
    metric_args: List[Dict[str, Any]] = [config["metrics"][met] for met in config["metrics"]]

    # Build optimizer, learning rate scheduler.
    # Delete every line containing lr_scheduler to disable scheduler.
    trainable_params: Iterable[Tensor] = filter(lambda p: p.requires_grad, model.parameters())
    optimizer: Optimizer = config.initialize("optimizer", torch.optim, trainable_params)

    lr_scheduler: Optional = config.initialize("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    trainer: Trainer = Trainer(
        model,
        loss_fn,
        loss_args,
        metric_fns,
        metric_args,
        optimizer,
        config=config,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    args: ArgumentParser = ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c", "--config", default=None, type=str, help="config file path (default: None)"
    )
    args.add_argument(
        "-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)"
    )
    args.add_argument(
        "-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)"
    )

    # Custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options: List[CustomArgs] = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target=("optimizer", "args", "lr")),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target=("data_loader", "args", "batch_size")
        ),
    ]

    config: ConfigParser = ConfigParser(args, options)
    main(config)
