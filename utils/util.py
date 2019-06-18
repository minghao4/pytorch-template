from collections import OrderedDict
from datetime import datetime, timedelta
from itertools import repeat
import json
from pathlib import Path
from typing import Any, Dict, IO, Union

from torch.utils.data import DataLoader


def ensure_dir(dirname: Union[Path, str]) -> None:
    """
    Checks if given directory exists, creates if it doesn't.

    Parameters
    ----------
    dirname : pathlib.Path or str
        The directory in question.
    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname: Path) -> None:
    """
    Reads JSON file.

    Parameters
    ----------
    fname : pathlib.Path
        File path to the JSON file in question.
    """
    handle: IO
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: Dict[str, Any], fname: Path) -> None:
    """
    Writes to JSON file.

    Parameters
    ----------
    content : dict of {str, Any}
        Formatted JSON output contents.

    fname : pathlib.Path
        File path for the output JSON file.
    """
    handle: IO
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader: DataLoader):
    """
    Wrapper function for endless data loader.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The torch `DataLoader` in question.

    Returns
    -------
    [TODO]
    """
    # TODO: figure out how to type hint loader and this fn's return.
    for loader in repeat(data_loader):
        yield from loader


class Timer:
    """
    Timer object.

    Attributes
    ----------
    cache : datetime.datetime
        The last recorded datetime.

    Methods
    -------
    check()
        Checks how much has elapsed.

    reset()
        Resets the cache.
    """
    def __init__(self) -> None:
        self.cache: datetime = datetime.now()

    def check(self) -> float:
        """
        Check how much time has elapsed. Resets cache to current time.

        Returns
        -------
        float
            The elapsed time in seconds.
        """
        now: datetime = datetime.now()
        duration: timedelta = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self) -> None:
        """Reset cache."""
        self.cache = datetime.now()
