import logging

import pandas as pd
from .classes import Mask
from loguru import logger

def save_to_csv(data: Mask) -> None:
    rows = []

    parent_dict = data.__dict__
    grains_dict = parent_dict.pop("grains")

    for grain_num, grain in grains_dict.items():
        row = {
            **grain.__dict__,
            **parent_dict
        }

        # Remove datatypes unfit for csv
        for key in ("dir", "mask", "config", "mask_rgb"):
            row.pop(key, None)

        rows.append(row)

    output_filename = f"output/{data.filename}/{data.filename}_statistics.csv"

    df = pd.DataFrame(rows)
    df.to_csv(output_filename, index=False)

    logger.info(
            f"exported statistics to {output_filename}",
        )
