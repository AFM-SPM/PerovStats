# Data Flow

## Input
- PerovStats takes AFM scan files as an input, as well as a config file (`default_config.yaml` by default) containing values for variables including the expected location of the files to process.

## Processing steps
The program will:
1. Loads the file and convert to an image mask
2. Performs a fourier transform to isolate the topograhy of the perovskite material
3. Optionally detect and remove smear areas from the high-pass so grains aren't detected there in error
4. Analyses this new image and generate a mask outlining the edges of grains
5. Counts and generates data about both individual grains and averages of all grains in the scan
6. Exports this data to `.csv` files along with a copy of the configuration options used in a `.yaml` file

## Output
All output data is by default saved to a sub-folder with the same name as the original file under an `/output/` directory (editable in the config).

The folder contains:
- An `images` folder of `.png` files for:
    - A `graphs` subfolder containing:
        - grain areas histogram
        - grain circularity rating histogram
    - The highpassed image (isolated perovskite grains)
    - The highpassed image with the final mask overlayed
    - The lowpassed image (removed silicon grains)
    - The final mask (black background)
    - The original image
    - The original image with the final mask overlayed
    - Grains coloured in
    - Un-coloured grains with detected means marked in red
- A copy of the config settings used to generate the data
- `.csv` files containing the statistics of the image as a whole and individual grains:
    - `grain_statistics` - one row per grain, individual stats
    - `image_statistics` - one row total, overall data of image and all grains within it

### Output directory stucture
```text
output/
├─ [input_filename]/
│  ├─ images/
│  │  ├─ graphs/
│  │  │  ├─ [input_filename]_grain_areas_hist.png
│  │  │  └─ [input_filename]_grain_circularity_hist.png
│  |  ├─ [input_filename]_highpass_mask_overlay.png
│  │  ├─ [input_filename]_highpass.png
│  │  ├─ [input_filename]_lowpass.png
│  │  ├─ [input_filename]_mask.png
│  │  ├─ [input_filename]_original_mask_overlay.png
│  │  ├─ [input_filename]_original.png
│  │  ├─ [input_filename]_rgb_grains.png
│  │  └─ [input_filename]_smears.png
│  ├─ config.yaml
│  ├─ grain_statistics.csv
│  └─ image_statistics.csv
```
