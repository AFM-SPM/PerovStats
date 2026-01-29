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
import logging
import sys
import copy
from pathlib import Path
from importlib import resources
from argparse import ArgumentParser
from argparse import Namespace
from argparse import RawDescriptionHelpFormatter

from yaml import safe_load
from topostats.filters import Filters
from topostats.io import LoadScans
import pandas as pd
from loguru import logger

from .grains import find_grains
from .fourier import create_masks
from .statistics import save_to_csv, save_config
from .classes import ImageData, PerovStats

LOGGER = logging.getLogger(__name__)


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


def main(args: list[str] | None = None) -> None:
    """
    Entrypoint for perovstats, calls main functions in the process.

    Parameters
    ----------
    args : list[str], optional
        Arguments.
    """
    logging.basicConfig(filename="perovstats.log", level=logging.INFO)
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

    print("fs config freq cutoff:", fs_config["cutoff_freq_nm"])

    # Update from command line arguments if specified
    fs_config.update({k: v for k, v in vars(args).items() if v is not None})

    print("fs config freq cutoff AFTER:", fs_config["cutoff_freq_nm"])

    cutoff = fs_config.get("cutoff")
    cutoff_freq_nm = fs_config.get("cutoff_freq_nm")

    if not (cutoff or cutoff_freq_nm):
        msg = "Must supply either `cutoff` or `cutoff_freq_nm`"
        raise ValueError(msg)

    # Non-recursively find files
    base_dir = get_arg("base_dir", args, config, "./")
    file_ext = get_arg("file_ext", args, config, "")
    img_files = list(Path(base_dir).glob("*" + file_ext))

    # Get / make output_dir
    output_dir = get_arg("output_dir", args, config, "./output")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_config = config["loading"]
    load_config["channel"] = get_arg("channel", args, load_config, "Height")

    # Load scans
    loadscans = LoadScans(img_files, **load_config)
    loadscans.get_data()
    image_dicts = loadscans.img_dict

    perovstats_object = PerovStats(config=config, images=[])
    for filename, topostats_object in image_dicts.items():
        image_data = ImageData(
            filename=filename,
            pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
            image_original=topostats_object["image_original"],
            image_flattened=None)
        perovstats_object.images.append(image_data)

    LOGGER.info("Loaded %s images", len(perovstats_object.images))

    filter_config = config["filter"]
    if filter_config["run"]:
        filter_config.pop("run")
        LOGGER.info("%s", filter_config)
        # apply filters
        for image_data in perovstats_object.images:
            filename = image_data.filename
            original_image = image_data.image_original
            pixel_to_nm_scaling = image_data.pixel_to_nm_scaling
            LOGGER.debug("[%s] Image dimensions: %s", filename, original_image.shape)
            LOGGER.info("[%s] : *** Filtering ***", filename)
            _filter_config = copy.deepcopy(filter_config)
            filters = Filters(
                image=original_image,
                filename=filename,
                pixel_to_nm_scaling=pixel_to_nm_scaling,
                **_filter_config,
            )
            image_data.pixel_to_nm_scaling = pixel_to_nm_scaling
            filters.filter_image()

    # Apply fourier analysis and create binary mask of resultant high-pass image
    create_masks(perovstats_object)

    # Find grains from mask
    find_grains(perovstats_object)

    for image_object in perovstats_object.images:
        # Save image and grain data to their own .csv file
        image_df = pd.DataFrame([image_object.to_dict()])
        grains_list = []
        for grain in image_object.grains.values():
            grains_list.append(grain.to_dict())
        grain_df = pd.DataFrame(grains_list)

        output_filename = f"{output_dir}/{image_object.filename}/image_statistics.csv"
        save_to_csv(image_df, output_filename)

        output_filename = f"{output_dir}/{image_object.filename}/grain_statistics.csv"
        save_to_csv(grain_df, output_filename)

        # Save the config settings in a .yaml
        output_filename = Path(output_dir) / Path(image_object.filename) / "config.yaml"
        save_config(perovstats_object.config, output_filename)

        logger.info(
            f"exported to {output_filename}",
        )
