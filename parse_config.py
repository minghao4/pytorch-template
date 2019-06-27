from argparse import ArgumentParser, Namespace
from datetime import datetime
from functools import reduce
import logging
from logging import Logger
from operator import getitem
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, NamedTuple, Optional, Union

from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    """
    Configuration JSON parser object.

    Attributes
    ----------
    config : dict of {str, Any}
        Dictionary of properties in json file.

    cfg_fname : pathlib.Path
        Config json file path.

    log_dir : pathlib.Path
        Directory path to save logs to.

    log_levels : int
        Verbosity of logging.

    resume : pathlib.Path
        Config json to resume from.

    save_dir : pathlib.Path
        Directory path to save to.

    Methods
    -------
    initialize()
        Initialize object instances from handle and args.

    get_logger()
        Produce a logger based on given verbosity level and logging configuration file.
    """
    def __init__(
        self,
        args: Union[ArgumentParser, Namespace],
        options: Union[List[NamedTuple], str] = "",
        timestamp: bool = True,
    ) -> None:
        # Parse default and custom cli options.
        opt: Union[NamedTuple, str]
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args()

        if args.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume:
            self.resume: Optional[Path] = Path(args.resume)
            self.cfg_fname: Path = self.resume.parent / "config.json"
        else:
            msg_no_cfg: str = \
                "Configuration file need to be specified. Add '-c config.json', for example."

            assert args.config is not None, msg_no_cfg
            self.resume = None
            self.cfg_fname = Path(args.config)

        # Load config file and apply custom cli options.
        config: Dict[str, Any] = read_json(self.cfg_fname)
        self._config: Dict[str, Any] = _update_config(config, options, args)

        # set save_dir where trained model and log will be saved.
        save_dir: Path = Path(self.config["trainer"]["save_dir"])
        timestamp: str = datetime.now().strftime(r"%m%d_%H%M%S") if timestamp else ""

        exper_name: str = self.config["name"]
        self._save_dir: Path = save_dir / "models" / exper_name / timestamp
        self._log_dir: Path = save_dir / "log" / exper_name / timestamp

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / "config.json")

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels: Dict[int, int] = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    def initialize(self, name: str, module: ModuleType, *args, **kwargs) -> object:
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding keyword args given as 'args'.

        Parameters
        ----------
        name : str
            Name of the object to initialize.

        module : types.ModuleType
            The module the object is located.

        Returns
        -------
        object
            The initialized object with the given args/kwargs.
        """
        module_name: str = self[name]["type"]
        module_args: Dict[str, Any] = dict(self[name]["args"])
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"

        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name: str) -> Any:
        return self.config[name]

    def get_logger(self, name: str, verbosity: int = 2) -> Logger:
        """
        Get the logger.

        Parameters
        ----------
        name : str
            The name of the logger

        verbosity : int, optional
            The verbosity of logging (default is 2)

        Returns
        -------
        logging.Logger
            The logger object.
        """
        msg_verbosity: str = "verbosity option {} is invalid. Valid options are {}.".format(
            verbosity, self.log_levels.keys()
        )
        assert verbosity in self.log_levels, msg_verbosity
        logger: Logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # Setting read-only attributes.
    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def save_dir(self) -> Path:
        return self._save_dir

    @property
    def log_dir(self) -> Path:
        return self._log_dir


# Helper functions used to update config dict with custom cli options.
def _update_config(
    config: Dict[str, Any], options: List[NamedTuple], args: Namespace
) -> Dict[str, Any]:
    """
    Update the configuration dictionary with custom cli args.

    Parameters
    ----------
    config : dict of {str, Any}
        Current configuration dictionary.

    options : list of train.CustomArgs
        List of custom arguments

    args : argparse.Namespace
        The arguments.

    Returns
    -------
    config : dict of {str, Any}
        The updated configuration dictionary.
    """
    opt: Union[NamedTuple, str]
    for opt in options:
        value: Optional = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)

    return config


def _get_opt_name(flags: List[str]) -> str:
    """
    Get the name of the optional argument.

    Parameters
    -----------
    flags : list of str
        The list of flags for the custom argument.

    Returns
    -------
    str
        The name of the custom argument.
    """
    flg: str
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")

    return flags[0].replace("--", "")


# TODO: Figure out type hintint for these 2 helpers.
def _set_by_path(tree, keys, value) -> None:
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
