# Data Flow

## Input
- PerovStats takes .spm files as an input, as well as a config file (`default_config.yaml` by default) containing values for variables including the expected location of the `.spm` files to process.

## Processing steps
The program will:
1. Loads the `.spm` file and convert to an image mask
2. Performs a fourier transform to isolate the topograhy of the perovskite material
3. Optionally detect and remove smear areas from the high-pass so grains aren't detected ther in error
4. Analyses this new image and generate a mask outlining the edges of grains
5. Counts and generates data about both individual grains and averages of all grains in the scan
6. Exports this data to `.csv` files along with a copy of the configuration options used in a `.yaml` file

## Output
All output data is by default saved to a sub-folder with the same name as the original `.spm` file under an `/output/` directory (editable in the config).

The folder contains:
- An `images` folder of `.jpg` files for:
    - Marked and coloured grains (coloured grains)
    - Post-fourier transform scan (high-pass)
    - Data removed from the fourier transofrm (low-pass)
    - High-pass image with the mask overlayed (mask overlay)
    - Grain outline mask (mask)
    - Original image (original)
- A copy of the config settings used to generate the data
- `.csv` files containing the statistics of the image as a whole and individual grains:
    - `grain_statistics` - one row per grain, individual stats
    - `image_statistics` - one row total, overall data of image and all grains within it

### Output directory stucture
```text
output/
├─ [spm_filename]/
│  ├─ images/
│  │  ├─ [spm_filename]_coloured_grains.jpg
│  │  ├─ [spm_filename]_high_pass.jpg
│  │  ├─ [spm_filename]_low_pass.jpg
│  |  ├─ [spm_filename]_mask_overlay.jpg
│  │  ├─ [spm_filename]_mask.jpg
│  │  └─ [spm_filename]_original.jpg
│  ├─ config.yaml
│  ├─ grain_statistics.csv
│  └─ image_statistics.csv
```
