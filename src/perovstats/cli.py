#
# Copyright: © 2025 University of Sheffield
#
# Authors:
#   Tamora James <t.d.james@sheffield.ac.uk>
#   Toby Allwood <t.allwood@sheffield.ac.uk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Command-line interface for PerovStats workflow."""

from __future__ import annotations
import sys
from pathlib import Path
from importlib import resources
from argparse import ArgumentParser
from argparse import Namespace
from argparse import RawDescriptionHelpFormatter

from loguru import logger
from yaml import safe_load

from .gui.app import PerovStatsApp
from .processing import process


def _parse_args(args: list[str]) -> Namespace:
    """
    Set up argument parser.

    Parameters
    ----------
    args : list[str]
        Command line arguments.

    Returns
    -------
    Namespace
        Argument data.
    """
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "-d",
        "--base_dir",
        type=str,
        default=None,
        help="Directory in which to search for data files",
    )
    parser.add_argument(
        "-e",
        "--file_ext",
        type=str,
        default=None,
        help="File extension of the data files",
    )
    parser.add_argument(
        "-n",
        "--channel",
        type=str,
        default=None,
        help="Name of data channel to use",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Directory to which to output results",
    )
    parser.add_argument(
        "-f",
        "--cutoff_freq_nm",
        type=float,
        help="Cutoff frequency in nm",
    )
    parser.add_argument(
        "-u",
        "--cutoff",
        type=float,
        default=0.05,
        help="Cutoff as proportion of Nyquist frequency",
    )
    parser.add_argument(
        "-w",
        "--edge_width",
        type=float,
        default=0.03,
        help="Edge width as proportion of Nyquist frequency",
    )
    return parser.parse_args(args)


def get_arg(key: str, args: Namespace, config: dict, default: str | None = None) -> str:
    """
    Get argument from namespace or configuration dictionary.

    Parameters
    ----------
    key : str
        Argument key.
    args : Namespace
        Arguments namespace.
    config : dict
        Configuration dictionary.
    default : str, optional
        Default value for argument.

    Returns
    -------
    str
        Argument value.
    """
    arg = vars(args)[key]
    if not arg:
        arg = config.get(key, default)
    return arg


def setup_logger() -> None:
    """
    Set up loguru, defining the output directory and filename and
    define the syntax for log messages.
    """
    logger.remove()
    logger.add("logs/PerovStats-{time:YYYY-MM-DD-HH-mm-ss}.log", level="DEBUG")
    logger.add(sys.stdout,
               level="DEBUG",
               format="<blue>{time:HH:mm:ss}</blue> | <level>{level: <7}</level> | <magenta>{file: <15}</magenta> | {message}",
               colorize=True)


@logger.catch
def main(args: list[str] | None = None) -> None:
    """
    Parse input args, load images from the selected directory and call the process()
    function to start the program.

    Parameters
    ----------
    args : list[str]
        List of arguments given in the command line when running PerovStats.
    """
    setup_logger()

    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)

    # read config file
    config_file_arg: str | None = args.config_file
    if config_file_arg:
        with Path(config_file_arg).open() as f:
            config = safe_load(f)
    else:
        with Path(resources.files(__package__) / "./default_config.yaml").open() as f:
            config = safe_load(f)

    fs_config = config.get("freqsplit", {})

    # Update from command line arguments if specified
    fs_config.update({k: v for k, v in vars(args).items() if v is not None})

    # Non-recursively find files
    base_dir = get_arg("base_dir", args, config, "./")
    file_ext = get_arg("file_ext", args, config, "")
    img_files = list(Path(base_dir).glob("*" + file_ext))

    # Get / make output_dir
    output_dir = get_arg("output_dir", args, config, "./output")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load_config = config["loading"]
    config["loading"]["channel"] = get_arg("channel", args, config["loading"], "Height")

    if config["gui"]["run"]:
        app = PerovStatsApp(process, img_files, config, output_dir)
        app.mainloop()
    else:
        process(img_files, config, output_dir)
