from pathlib import Path
from art import tprint
import time

from loguru import logger
from topostats.io import LoadScans
import pandas as pd

from .core.classes import ImageData, PerovStats
from .core.io import save_to_csv, save_config
from .segmentation import segment_image_cellpose
from .grains import find_grains
from .fourier import split_frequencies
from .smears import find_smear_areas
from .pruning import prune_mask


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
    loadscans = LoadScans(img_files, config)
    try:
        loadscans.get_data()
    except ValueError as e:
        logger.warning(e)
        logger.warning(f"Channel {load_config['channel']} not found in file. Please ensure the config option is correct and all files contain the required channel.")
    image_dicts = loadscans.img_dict

    # Create the dataclasses for the whole process and for each image found
    perovstats_object = PerovStats(config=config, images=[])
    for filename, topostats_object in image_dicts.items():
        image_data = ImageData(
            success=True,
            filename=filename,
            pixel_to_nm_scaling=topostats_object.pixel_to_nm_scaling,
            image_original=topostats_object.image_original,
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

        # Apply fourier transform to split the image into a low-passed and high-passed image
        split_frequencies(perovstats_object.config, image_object)

        # If frequency splitting was run and failed skip processing on the rest of the image
        if not image_object.success:
            continue

        # Generate grain mask of the high-passed image
        segment_image_cellpose(perovstats_object.config, image_object)

        # Remove small offshoots in the mask and connect sections with small breaks
        prune_mask(perovstats_object.config, image_object)

        # Find smear areas to be ignored/ removed
        find_smear_areas(perovstats_object.config, image_object)

        # Identify individual grains from mask and generate statistics on them
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
    time_taken : str
        The formatted total time taken on processing all given images.
    time_per_image : str
        The formatted average time taken per image during the process. This is required as a parameter
        as it must be calculated before the total time is formatted.
    """
    successful_no = sum(image.success for image in perovstats_object.images)
    success_perc = round((successful_no / len(perovstats_object.images)) * 100, 2)

    logger.success("Process completed successfully.")
    print("----------------------------------------------------------------------------------------------------\n")
    tprint("PerovStats", font="epic")
    print(
        f"----------------------------------------------------------------------------------------------------\n"
        f"Base Directory                        : {perovstats_object.config['base_dir']}\n"
        f"Output Directory                      : {perovstats_object.config['output_dir']}\n"
        f"File Extension                        : {perovstats_object.config['file_ext']}\n"
        f"Files Found                           : {len(perovstats_object.images)}\n"
        f"Successfully Processed                : {successful_no} ({success_perc}%)\n"
        f"Time Taken                            : {time_taken} (~{time_per_image}/image)\n"
        f"----------------------------------------------------------------------------------------------------"
    )


def format_time(seconds: float) -> str:
    """
    Turn the time recorded (float of seconds) into text formatted into hours, minutes,
    and seconds depending on the length.

    Parameters
    ----------
    seconds : float
        The overall seconds the process took from start to finish.

    Returns
    -------
    str
        The formatted time showing hours, minutes, and seconds.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {int(seconds)}s"

    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
