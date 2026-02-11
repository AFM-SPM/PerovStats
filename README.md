# PerovStats
An program to process AFM scans of perovskite and generate usable data and statistics.

## Installation

To install PerovStats directly from GitHub via ssh:

```console
pip install git@github.com:AFM-SPM/PerovStats.git
```

To install PerovStats by cloning from the GitHub repository:

```console
git clone https://github.com/AFM-SPM/PerovStats.git
cd PerovStats
pip install -r requirements.txt
```

## Documentation

### Basic usage

Run the command `perovstats` in the terminal.
- Uses `src/perovstats/default_config.yaml` for configuration options, below details custom arguments avaliable.

---

### Command-line interface

```console
usage: perovstats [-h] [-c CONFIG_FILE] [-d BASE_DIR] [-e FILE_EXT] [-n CHANNEL] [-o OUTPUT_DIR] [-w EDGE_WIDTH]

Command-line interface for PerovStats workflow.

options:
  -h, --help            show this help message and exit
  -c CONFIG_FILE, --config_file CONFIG_FILE
                        Path to configuration file
  -d BASE_DIR, --base_dir BASE_DIR
                        Directory in which to search for data files
  -e FILE_EXT, --file_ext FILE_EXT
                        File extension of the data files
  -n CHANNEL, --channel CHANNEL
                        Name of data channel to use
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Directory to which to output results
  -w EDGE_WIDTH, --edge_width EDGE_WIDTH
                        Edge width as proportion of Nyquist frequency
```

---

### Accepted file types

Currently PerovStats only handles `.spm` files generated from an AFM scan.

---

### Features

- Image frequency splitting to isolate perovskite topology (optional)
- Automatic segmentation and grain finding of an image
- Statistics and `.csv` generation about processed image, including:
  - Image-wide statistics
    - Number of grains found
    - Grains per nm^2
    - Pixel to nm scaling
    - Mean, median, and mode grain sizes
  - Per-grain statistics
    - Grain area
    - Grain circularity rating (0-1)
  - Metadata about values determined during processing such as the cutoff frequency (if frequency splitting was used) and the threshold used for grain finding

## License

This software is licensed under the [GPLv3](LICENSE).
