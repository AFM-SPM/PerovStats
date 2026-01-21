# Data Flow

## Input
- PerovStats takes .spm files as an input, as well as a config file (`default_config.yaml` by default) containing values for variables including the expected location of the `.spm` files to process.

## Processing steps
The program will:
1. Loads the `.spm` file and convert to an image mask
2. Optionally performs a fourier transform to isolate the topograhy of the perovskite material
3. Analyses this new image and generate a mask outlining the edges of grains
4. Counts and generates data about both individual grains and averages of all grains in the scan
5. Exports this data to a `.csv` file

## Output
All output data is saved to `/output/` (editable in the config) under a sub-folder with the same name as the original `.spm` file.

This folder contains:
- A copy of the config settings used to generate the data
- A `.jpg` of the post-fourier transform scan
- A `.jpg` of the grain outline mask
- The original image used
- `.csv` files containing the statistics of the image as a whole and individual grains

### Output directory stucture
```text
output/
├─ [spm_filename]/
│  ├─ images/
│  │  ├─ [spm_filename]_high_pass.jpg
│  |  ├─ [spm_filename]_mask.jpg
│  │  └─ [spm_filename]_original.jpg
│  ├─ config.yaml
│  ├─ grain_statistics.csv
│  └─ image_statistics.csv
```
