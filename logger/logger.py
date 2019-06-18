import logging
import logging.config
from pathlib import Path
from typing import Any, Dict, Union

from utils import read_json


def setup_logging(
    save_dir: Path,
    log_config: Union[Path, str] = "logger/logger_config.json",
    default_level: int = logging.INFO,
) -> None:
    """
    Setup logging configuration.

    Parameters
    ----------
    save_dir : pathlib.Path
        The directory path to save files to.

    log_config : pathlib.Path or str
        The file path of the logging configuration JSON.

    default_level : int, optional
        Default logging level (default is logging.INFO).
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config: Dict[str, Any] = read_json(log_config)

        # Modify logging paths based on the run config.
        _: str
        handler: Dict[str, Any]
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])

        logging.config.dictConfig(config)

    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
