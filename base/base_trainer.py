from abc import abstractmethod
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

from numpy import inf
import torch
from torch import device
from torch.nn import DataParallel, Module
from torch.optim import Optimizer

from logger import TensorboardWriter
from parse_config import ConfigParser


class BaseTrainer:
    """
    Base class for all trainers.

    Attributes
    ----------
    model : torch.nn.Module
        The model.

    loss_fn : callable
        Loss function.

    loss_args : dict of {str, Any}
        Keyword arguments of the loss function.

    metric_fns : list of callable
        List of metric functions.

    metric_args : list of dict of {str, Any}
        List of keyword arguments of the metric functions, matched by index.

    optimizer : torch.optimizer.Optimizer
        The optimizer.

    config : parse_config.ConfigParser
        The config parsing object.

    Methods
    -------
    train()
        Full training logic.
    """

    def __init__(
        self,
        model: Module,
        loss_fn: Callable,
        loss_args: Dict[str, Any],
        metric_fns: List[Callable],
        metric_args: List[Dict[str, Any]],
        optimizer: Optimizer,
        config: ConfigParser,
    ):
        self.config: ConfigParser = config
        self.logger: Logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        # Setup GPU device if available.
        self.device: str
        device_ids: List[int]
        self.device, device_ids = self._prepare_device(config["n_gpu"])

        # Move model into configured device(s).
        self.model: Module = model.to(self.device)
        if len(device_ids) > 1:
            self.model = DataParallel(model, device_ids=device_ids)

        # Set loss function and arguments.
        self.loss_fn: Callable = loss_fn
        self.loss_args: Dict[str, Any] = loss_args

        # Set all metric functions and associated arguments.
        self.metric_fns: List[Callable] = metric_fns
        self.metric_args: List[Dict[str, Any]] = metric_args

        # Set optimizer.
        self.optimizer: Optimizer = optimizer

        # Set training configuration.
        cfg_trainer: Dict[str, Any] = config["trainer"]
        self.epochs: int = cfg_trainer["epochs"]
        self.save_period: int = cfg_trainer["save_period"]
        self.monitor: str = cfg_trainer.get("monitor", "off")

        # Configuration to monitor model performance and save best.
        if self.monitor == "off":
            self.mnt_mode: str = "off"
            self.mnt_best: float = 0
        else:
            self.mnt_metric: str
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop: float = cfg_trainer.get("early_stop", inf)

        self.start_epoch: int = 1
        self.checkpoint_dir: Path = config.save_dir

        # Setup visualization writer instance.
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer["tensorboard"])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch: int) -> dict:
        """
        TODO: check if type-hinting is correct for abstractmethod.

        Training logic for an epoch. If not implemented in child class, raise `NotImplementedError`.

        Parameters
        ----------
        epoch : int
            The current epoch.

        Returns
        -------
        dict
            A dictionary containing the logged information.
        """
        raise NotImplementedError

    def train(self) -> None:
        """Full training logic."""
        for epoch in range(self.start_epoch, self.epochs + 1):
            result: dict = self._train_epoch(epoch)

            # Save logged information in log dict.
            log: Dict[str, float] = {"epoch": epoch}
            key: str
            value: Union[float, List[float]]
            for key, value in result.items():
                if key == "metrics":
                    assert isinstance(value, List[float])
                    i: int
                    mtr: str
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})

                elif key == "val_metrics":
                    assert isinstance(value, List[float])
                    log.update(
                        {"val_" + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)}
                    )

                else:
                    assert isinstance(value, float)
                    log[key] = value

            # Print logged info to stdout.
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # Evaluate model performance in accordance with the configured metric, save best
            # checkpoint as model_best.
            best: bool = False
            if self.mnt_mode != "off":
                try:
                    # Check whether model performance improved or not, according to specified
                    # metric(mnt_metric).
                    improved: bool = (
                        self.mnt_mode == "min" and log[self.mnt_metric] <= self.mnt_best
                    ) or (self.mnt_mode == "max" and log[self.mnt_metric] >= self.mnt_best)

                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found.\n".format(self.mnt_metric)
                        + "Model performance monitoring is disabled."
                    )
                    self.mnt_mode = "off"
                    improved = False
                    not_improved_count: int = 0

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} ".format(self.early_stop)
                        + "epochs.\nTraining stops."
                    )
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _prepare_device(self, n_gpu_use: int) -> Tuple[device, List[int]]:
        """
        Setup GPU device if available, move model into configured device.

        Parameters
        ----------
        n_gpu_use : int
            The number of GPUs to use.

        Returns
        -------
        tuple
            A tuple of the device in use and a list of device IDs.
        """
        n_gpu: int = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There's no GPU available on this machine, "
                + "training will be performed on CPU."
            )
            n_gpu_use = 0

        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU's configured to use is {}, ".format(n_gpu_use)
                + "but only {} are available on this machine.".format(n_gpu)
            )
            n_gpu_use = n_gpu

        device: device = device("cuda:0" if n_gpu_use > 0 else "cpu")
        list_ids: List[str] = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch: int, save_best: bool = False) -> None:
        """
        Saving current state as a checkpoint.

        Parameters
        ----------
        epoch : int
            The current epoch.

        save_best : bool
            If True, the saved checkpoint is renamed to "model_best.pth"
        """
        arch: str = type(self.model).__name__
        state: Dict[str, Any] = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename: str = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        torch.save(state, filename)

        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path: str = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path: Path) -> None:
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path: str = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint: dict = torch.load(resume_path)

        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # Load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that of"
                + " checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # Load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint["config"]["optimizer"]["type"] != self.config["optimizer"]["type"]:
            self.logger.warning(
                "Warning: Optimizer type given in config file is different from that of checkpoint."
                + " Optimizer parameters not being resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
