# PerovStats
An program to process AFM scans of perovskite and generate usable data and statistics from them

## Installation

To install PerovStats directly from GitHub via ssh:

```console
pip install git@github.com:AFM-SPM/PerovStats.git
```

To install PerovStats by cloning from the GitHub repository:

```console
git clone https://github.com/AFM-SPM/PerovStats.git
cd PerovStats
python -m pip install .
```

## Documentation

### Basic usage

Run the command `perovstats` in the terminal.
- Uses `src/perovstats/default_config.yaml` for configuration options, below details custom arguments avaliable.

### Command-line interface

```console
usage: perovstats [-h] [-c CONFIG_FILE] [-d BASE_DIR] [-e FILE_EXT] [-n CHANNEL] [-o OUTPUT_DIR] [-f CUTOFF_FREQ_NM]
                 [-u CUTOFF] [-w EDGE_WIDTH]

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
  -f CUTOFF_FREQ_NM, --cutoff_freq_nm CUTOFF_FREQ_NM
                        Cutoff frequency in nm
  -u CUTOFF, --cutoff CUTOFF
                        Cutoff as proportion of Nyquist frequency
  -w EDGE_WIDTH, --edge_width EDGE_WIDTH
                        Edge width as proportion of Nyquist frequency
```

## License

This software is licensed under the [GPLv3](LICENSE).
