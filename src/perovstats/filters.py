import copy

from loguru import logger
from topostats.filters import Filters

from .core.classes import ImageData

def run_filters(
    config: dict[str, any],
    image_object: ImageData
) -> None:
    """
    Run the TopoStats Filters module.

    Parameters
    ----------
    config : dict[str, any]
        Config options given at the start of the program.
    image_object : ImageData
        An ImageData object for the current image to be filtered.
    """
    filter_config = config["filter"]
    if filter_config["run"]:
        filename = image_object.filename
        logger.info(f"[{filename}] : *** Filtering ***")
        # apply filters
        _filter_config = copy.deepcopy(filter_config)
        _filter_config.pop("run")
        filters = Filters(
            image=image_object.image_original,
            filename=filename,
            pixel_to_nm_scaling=image_object.pixel_to_nm_scaling,
            **_filter_config,
        )
        filters.filter_image()
