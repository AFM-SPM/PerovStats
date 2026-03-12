from pathlib import Path
from art import tprint
import time

from loguru import logger
from topostats.io import LoadScans
import pandas as pd

from .core.io import save_to_csv, save_config
from .core.classes import ImageData, PerovStats
from .filters import run_filters
from .grains import find_grains
from .fourier import run_frequency_splitting
from .smears import find_smear_areas


def process(
        img_files,
        config,
        output_dir
    ) -> None:
    """
    Main method for running processes in PerovStats, calls functions
    for each section in turn and then saves final data.

    Parameters
    ----------
    img_files : List[Path]
        List of paths to the image files found for processing.
    config : dict[str, any]
        All configuration options for running PerovStats.
    output_dir : Path
        Filepath of the directory to save images/ files to.
    """
    time_start = time.perf_counter()

    # Load scans
    load_config = config["loading"]
    loadscans = LoadScans(img_files, **load_config)
    try:
        loadscans.get_data()
    except ValueError as e:
        logger.warning(e)
        logger.warning(f"Channel {load_config['channel']} not found in file. Please ensure the config option is correct and all files contain the required channel.")
    image_dicts = loadscans.img_dict

    perovstats_object = PerovStats(config=config, images=[])
    for filename, topostats_object in image_dicts.items():
        image_data = ImageData(
            filename=filename,
            pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
            image_original=topostats_object["image_original"],
            image_flattened=None)
        perovstats_object.images.append(image_data)

    logger.info("----------------------------------------------------------")
    logger.info(f"Loaded {len(perovstats_object.images)} images")

    for idx, image_object in enumerate(perovstats_object.images):
        logger.info("----------------------------------------------------------")
        logger.info(f"Processing {image_object.filename} ({idx+1}/{len(perovstats_object.images)})")
        logger.info("----------------------------------------------------------")
        logger.debug(f"[{image_object.filename}] : Image dimensions: {image_object.image_original.shape}")
        logger.debug(f"[{image_object.filename}] : pixel_to_nm_scaling: {image_object.pixel_to_nm_scaling}")

        # Filter images
        run_filters(perovstats_object.config, image_object)

        # Apply fourier analysis and create binary mask of resultant high-pass image
        run_frequency_splitting(perovstats_object.config, image_object)

        # Find smear areas to be ignored/ removed
        find_smear_areas(perovstats_object.config, image_object)

        # Find grains from mask
        find_grains(perovstats_object.config, image_object)

        logger.info(f"[{image_object.filename}] : *** Exporting data ***")
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
            f"[{image_object.filename}] : Exported data and config to {Path(output_dir) / Path(image_object.filename)}",
        )

    time_end = time.perf_counter()
    time_taken = format_time(time_end - time_start)
    time_per_image = format_time((time_end - time_start) / len(perovstats_object.images))
    completion_message(perovstats_object, time_taken, time_per_image)


def completion_message(perovstats_object: PerovStats, time_taken: str, time_per_image: str) -> None:
    """
    Message to be printed at the end of a successful PerovStats run, includes basic config info for the user.

    Parameters
    ----------
    perovstats_object : PerovStats
        The main PerovStats class object containing all config options and input/ processed data.
    """
    logger.success("Process completed successfully.")
    print("----------------------------------------------------------------------------------------------------\n")
    tprint("PerovStats", font="epic")
    print(
        f"----------------------------------------------------------------------------------------------------\n"
        f"Base Directory                        : {perovstats_object.config['base_dir']}\n"
        f"Output Directory                      : {perovstats_object.config['output_dir']}\n"
        f"File Extension                        : {perovstats_object.config['file_ext']}\n"
        f"Files Found                           : {len(perovstats_object.images)}\n"
        f"Time Taken                            : {time_taken} (~{time_per_image}/image)\n"
        f"----------------------------------------------------------------------------------------------------"
    )


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {int(seconds)}s"

    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
