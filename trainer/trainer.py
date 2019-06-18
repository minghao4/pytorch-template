from typing import Any, Callable, Dict, List, Optional

import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from base import BaseTrainer
from parse_config import ConfigParser
from utils import inf_loop


class Trainer(BaseTrainer):
    """
    Trainer class.

    Attributes
    ----------
    config : parse_config.ConfigParser
        Parsed config file.

    data_loader : torch.utils.data.DataLoader
        Training set data loader.

    do_validation : bool
        Whether or not to perform validation.

    len_epoch : int
        Number of epochs.

    log_step : int
        Logging step.

    lr_scheduler : object or None
        Learning rate scheduler object.

    valid_data_loader : torch.utils.data.DataLoader or None
        Validation set data loader.

    Note:
        Inherited from BaseTrainer.
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
        data_loader: DataLoader,
        valid_data_loader: Optional[DataLoader] = None,
        lr_scheduler: Optional = None,
        len_epoch: Optional[int] = None,
    ) -> None:

        super().__init__(model, loss_fn, loss_args, metric_fns, metric_args, optimizer, config)
        self.config: ConfigParser = config  # TODO: isn't this in BaseTrainer already?
        self.data_loader: DataLoader = data_loader

        # Epoch-based training.
        if len_epoch is None:
            self.len_epoch: int = len(self.data_loader)

        # Iteration-based training.
        else:
            self.data_loader: DataLoader = inf_loop(data_loader)
            self.len_epoch: int = len_epoch

        self.valid_data_loader: Optional[DataLoader] = valid_data_loader
        self.do_validation: bool = self.valid_data_loader is not None
        self.lr_scheduler: Optional = lr_scheduler
        self.log_step: int = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output: Tensor, target: Tensor) -> ndarray:
        """
        Evaluate all metrics.

        Parameters
        ----------
        output : torch.Tensor
            Output tensor.

        target : torch.Tensor
            Target tensor.

        Returns
        -------
        np.ndarray
            Array of all accumulated metrics.
        """
        acc_metrics: ndarray = np.zeros(len(self.metrics))

        i: int
        metric: Callable
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target, **self.metric_args[i])
            self.writer.add_scalar("{}".format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch: int) -> Dict[str, Any]:
        """
        Training logic for a single epoch.

        Parameters
        ----------
        epoch : int
            current training epoch.

        Returns
        -------
        dict of {str, Any}
            Logged info.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss: float = 0
        total_metrics: ndarray = np.zeros(len(self.metrics))

        batch_idx: int
        data: Tensor
        target: Tensor
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output: Tensor = self.model(data)
            loss: Tensor = self.loss(output, target, **self.loss_args)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar("loss", loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss.item()
                    )
                )
                self.writer.add_image("input", make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        log: Dict[str, Any] = {
            "loss": total_loss / self.len_epoch,
            "metrics": (total_metrics / self.len_epoch).tolist(),
        }

        if self.do_validation:
            val_log: Dict[str, Any] = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch: int) -> Dict[str, Any]:
        """
        Validate after training an epoch.

        Parameters
        ----------
        epoch : int
            Current epoch.

        Returns
        -------
        dict of {str, Any}
            Logged info.

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss: int = 0
        total_val_metrics: ndarray = np.zeros(len(self.metrics))

        # no_grad is turned on when not training.
        with torch.no_grad():
            batch_idx: int
            data: Tensor
            target: Tensor
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output: Tensor = self.model(data)
                loss: Tensor = self.loss(output, target, **self.loss_args)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid")
                self.writer.add_scalar("loss", loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                self.writer.add_image("input", make_grid(data.cpu(), nrow=8, normalize=True))

        # Add histogram of model parameters to the TensorBoard.
        name: str
        p: Tensor
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")

        return {
            "val_loss": total_val_loss / len(self.valid_data_loader),
            "val_metrics": (total_val_metrics / len(self.valid_data_loader)).tolist(),
        }

    def _progress(self, batch_idx: int) -> str:
        """
        Progress bar.

        Parameters
        ----------
        batch_idx : int
            Current batch index.

        Returns
        -------
        str
            Current progress bar.
        """
        base: str = "[{}/{} ({:.0f}%)]"

        if hasattr(self.data_loader, "n_samples"):
            current: int = batch_idx * self.data_loader.batch_size
            total: int = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch

        return base.format(current, total, 100.0 * current / total)
