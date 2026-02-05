import copy

from loguru import logger

from topostats.filters import Filters

def run_filters(config, image_object):
    filter_config = config["filter"]
    if filter_config["run"]:
        # apply filters
        filename = image_object.filename
        original_image = image_object.image_original
        pixel_to_nm_scaling = image_object.pixel_to_nm_scaling
        logger.info(f"[{filename}] : *** Filtering ***")
        _filter_config = copy.deepcopy(filter_config)
        _filter_config.pop("run")
        filters = Filters(
            image=original_image,
            filename=filename,
            pixel_to_nm_scaling=pixel_to_nm_scaling,
            **_filter_config,
        )
        image_object.pixel_to_nm_scaling = pixel_to_nm_scaling
        filters.filter_image()
