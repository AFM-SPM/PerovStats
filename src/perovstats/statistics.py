import logging
from pathlib import Path
from yaml import safe_dump

import pandas as pd
from .classes import ImageData
from loguru import logger


def save_to_csv(df: pd.DataFrame, output_filename: str) -> None:
    df.to_csv(output_filename, index=False)
    logger.info(
        f"exported statistics to {output_filename} along with its configuration settings.",
    )


def save_config(config, output_filename) -> None:
    with (output_filename).open("w") as outfile:
        safe_dump(config, outfile, default_flow_style=False)
