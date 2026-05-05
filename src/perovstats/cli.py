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
import yaml

from loguru import logger
from yaml import safe_load

from .processing import run_process
from .config import update_module


def create_parser() -> ArgumentParser:
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
        "-s",
        "--segmentation",
        type=str,
        default=None,
        help='Method for segmenting an image into grains. Options: "traditional", "cellpose"'
    )

    subparsers = parser.add_subparsers(title="program", description="Available programs, listed below:", dest="module")

    # Create a sub-parsers for different stages of processing and tasks
    process_parser = subparsers.add_parser(
        "process",
        description="Process AFM images. Additional arguments over-ride defaults or those in the configuration file.",
        help="Process AFM images. Additional arguments over-ride defaults or those in the configuration file.",
    )

    # Run the relevant function with the arguments
    process_parser.set_defaults(func=process)

    return parser


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



def entry_point(manually_provided_args=None, testing=False) -> None:
    """
    Entry point for all PerovStats programs.

    Main entry point for running 'perovstats' which allows the different processing steps ('process', 'filter',
    'create_config' etc.) to be run.

    Parameters
    ----------
    manually_provided_args : None
        Manually provided arguments.
    testing : bool
        Whether testing is being carried out.

    Returns
    -------
    None
        Does not return anything.
    """

    # Parse command line options, load config (or default) and update with command line options
    parser = create_parser()
    args = parser.parse_args() if manually_provided_args is None else parser.parse_args(manually_provided_args)

    # No program specified, print help and exit
    if not args.module:
        parser.print_help()
        sys.exit()
    else:
        update_module(args=args)

    if testing:
        return args

    # call the relevant function
    args.func(args)

    return None


def deep_merge(base, custom):
    """
    Method for inserting custom config values into the full default config.
    """
    for key, value in custom.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


@logger.catch
def process(args: list[str] | None = None) -> None:
    """
    Parse input args, load images from the selected directory and call the process()
    function to start the program.

    Parameters
    ----------
    args : list[str]
        List of arguments given in the command line when running PerovStats.
    """
    setup_logger()

    # read config file - integrating custom config options into the default config if one was selected
    config_file_arg: str | None = args.config_file
    if config_file_arg:
        if Path(config_file_arg).exists():
            with Path(config_file_arg).open() as f:
                config = safe_load(f)
                default_config_path = Path("../src/perovstats/default_config.yaml")
                with default_config_path.open() as f:
                    config = yaml.safe_load(f)

                if config_file_arg.resolve() != default_config_path.resolve():
                    if config_file_arg.exists():
                        with config_file_arg.open() as f:
                            custom_config = yaml.safe_load(f)

                        config = deep_merge(config, custom_config)
        else:
            logger.error("Error: custom configuration file could not be found. Please check the set filepath above.")
    else:
        with Path(resources.files(__package__) / "./default_config.yaml").open() as f:
            config = safe_load(f)

    # Fourier-specific command line arguments
    fourier_keys = {'cutoff_freq_nm', 'cutoff', 'edge_width'}

    cli_args = {k: v for k, v in vars(args).items() if v is not None}

    # Assign fourier-specific command line arguments to the fourier subsection of config
    if 'fourier' not in config:
        config['fourier'] = {}
    for k in fourier_keys:
        if k in cli_args:
            config['fourier'][k] = cli_args.pop(k)

    if 'segmentation' not in config:
        config['segmentation'] = {}
    if 'segmentation' in cli_args:
        config['segmentation']['segmentation_method'] = cli_args.pop('segmentation')

    # Update top level config from remaining command line arguments
    config.update(cli_args)

    # Non-recursively find files
    base_dir = get_arg("base_dir", args, config, "./")
    file_ext = get_arg("file_ext", args, config, "")
    img_files = list(Path(base_dir).glob("*" + file_ext))

    # Get / make output_dir
    output_dir = get_arg("output_dir", args, config, "./output")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config["loading"]["channel"] = get_arg("channel", args, config["loading"], "Height")

    config["func"] = 'process'

    run_process(img_files, config, output_dir)
