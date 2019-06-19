import importlib
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Optional, Set, Union

from tensorboardX import SummaryWriter

from utils import Timer


class TensorboardWriter:
    """
    Object to write to Tensorboard.

    Attributes
    ----------
    mode : str
        Training or testing mode.

    selected_module : str
        The module the tb writer object is from.

    step : int
        How frequently to write.

    tag_mode_exceptions : set of str
        Tensorboard writer functions with exceptions with regards to tag and mode.

    tb_writer_ftns : set of str
        Tensorboard writer functions.

    timer : utils.Timer
        Timer.

    writer : torch.utils.tensorboard.SummaryWriter or tensorboardX.SummaryWriter or None
        The tensorboard writer object.

    Methods
    -------
    set_step()
        Sets the recording steps and checks timer.
    """
    def __init__(self, log_dir: Union[Path, str], logger: Logger, enabled: bool) -> None:
        self.writer: Optional[SummaryWriter] = None
        self.selected_module: str = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded: bool = False
            module: str
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    # mypy linting not happy here, but that's what the try-catch is for
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break

                except ImportError:
                    succeeded = False

                self.selected_module = module

            if not succeeded:
                message = (
                    "Warning: visualization (Tensorboard) is configured to use, but currently not "
                    + "installed on this machine. Please install either TensorboardX with "
                    + "'pip install tensorboardx', upgrade PyTorch to version >= 1.1 for using"
                    + "'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                )
                logger.warning(message)

        self.step: int = 0
        self.mode: str = ""

        self.tb_writer_ftns: Set[str] = {
            "add_scalar",
            "add_scalars",
            "add_image",
            "add_images",
            "add_audio",
            "add_text",
            "add_histogram",
            "add_pr_curve",
            "add_embedding",
        }
        self.tag_mode_exceptions: Set[str] = {"add_histogram", "add_embedding"}

        self.timer: Timer = Timer()

    def set_step(self, step: int, mode: str = "train") -> None:
        """
        Sets the recording steps.

        Parameters
        ----------
        step : int
            Current step.

        mode : str
            Training/testing mode.
        """
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer.reset()
        else:
            duration: float = self.timer.check()
            self.add_scalar("steps_per_sec", 1 / duration)

    def __getattr__(self, name: str) -> Callable:
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag: str, data: Any, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = "{}/{}".format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)

            return wrapper

        # Default action for returning methods defined in this class, set_step() for instance.
        else:
            try:
                # mypy linting issue but that's what try-catch is for
                attr: Callable = object.__getattr__(name)

            except AttributeError:
                raise AttributeError(
                    "type object '{}' has no attribute '{}'".format(self.selected_module, name)
                )

            return attr
