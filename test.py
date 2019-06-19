from argparse import ArgumentParser
from logging import Logger
from tqdm import tqdm
from typing import Any, Callable, Dict, List

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser


def main(config: ConfigParser) -> None:
    """
    Main testing function.

    Parameters
    ----------
    config : parse_config.ConfigParser
        Parsed configuration JSON file.
    """
    logger: Logger = config.get_logger("test")

    # Setup data_loader instance.
    data_loader: DataLoader = getattr(module_data, config["data_loader"]["type"])(
        config["data_loader"]["args"]["data_dir"],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2,
    )

    # Build model architecture.
    model: Module = config.initialize("arch", module_arch)
    logger.info(model)

    # Get function handles of loss and metrics as well as args.
    loss_fn: Callable = getattr(module_loss, config["loss"]["type"])
    loss_args: Dict[str, Any] = config["loss"]["args"]
    metric_fns: List[Callable] = [getattr(module_metric, met) for met in config["metrics"]]
    metric_args: List[Dict[str, Any]] = [config["metrics"][met] for met in config["metrics"]]

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint: dict = torch.load(config.resume)
    state_dict: dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # Prepare model for testing.
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    total_loss: float = 0.0
    total_metrics: Tensor = torch.zeros(len(metric_fns))

    with torch.no_grad():
        i: int
        data: Tensor
        target: Tensor
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output: Tensor = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss: Tensor = loss_fn(output, target, **loss_args)
            batch_size: int = data.shape[0]
            total_loss += loss.item() * batch_size

            j: int
            metric: Callable
            for j, metric in enumerate(metric_fns):
                total_metrics[j] += metric(output, target, **metric_args[j]) * batch_size

    n_samples: int = len(data_loader.sampler)
    log: Dict[str, Any] = {"loss": total_loss / n_samples}

    met: Callable
    log.update(
        {met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)}
    )

    logger.info(log)


if __name__ == "__main__":
    args: ArgumentParser = ArgumentParser(description="PyTorch Template")

    args.add_argument(
        "-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)"
    )
    args.add_argument(
        "-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)"
    )

    config: ConfigParser = ConfigParser(args)
    main(config)
