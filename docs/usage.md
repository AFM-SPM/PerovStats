# PerovStats usage instructions

This page assumes you have downloaded and set up PerovStats. If not, please follow the instructions found in [installation.md](installation.md).

If you have any issues please email t.allwood@sheffield.ac.uk - I will get back to you as soon as I can

## Running from the notebook

### [perovstats-demo.ipynb](../notebooks/perovstats-demo.ipynb)

#### This is the demo notebook for an explanation on running PerovStats and the processes it goes through. Use this notebook first to see how AFM files will be processed. As this is for example purposes, only one file can be inputted per run.

- The first code block is for imports and can be ignored. In the second code block there are paths for you to edit:
    - `img_file`

        The input filepath (.spm or other). Two files are provided for you to test, simply remove the `# ` from one and add it to the start of the other to switch files. Alternatively you can add your own filepath to run the program on.
    - `output_dir`

        The directory to save the results and output to. If the folder does not exist this will be created while running PerovStats.
    - `config_path`

        The configuration file (`.yaml`) to be used in the program's run. For the demo notebook this can be left as is and default configuration options will be used.

        (For help creating a custom config file refer to the [config documentation](config.md))

- You can now start running the cells (one at a time or all at once). Images will appear through the notebook as cells run showing you the latest stage of the process. The segmentation cell may take a few minutes to complete, if no error message shows assume everything is running as expected.

- Images of the scan, graphs, `.csv` files and a copy of the configuration options used will be saved to the output directory you chose earlier.

### [perovstats-multi-process.ipynb](../notebooks/perovstats-multi-process.ipynb)

#### This is the main notebook to use when running PerovStats, and allows multiple files to be processed in one run

- The first code block is for imports and can be ignored. In the second code block there are paths for you to edit:
    - `img_files`

        The folder containing all images you want to process.
    - `output_dir`

        The directory to save the results and output to. If the folder does not exist this will be created while running PerovStats.
    - `config_path`

        The configuration file (`.yaml`) to be used in the program's run. For the demo notebook this can be left as is and default configuration options will be used.

- You can now start running the cells (one at a time or all at once). Once all input images have been loaded they will be looped through one by one and processed. You can see log messages on the progress of the program as it progresses down the notebook.

- Images of the scan, graphs, `.csv` files and a copy of the configuration options used will be saved to the output directory you chose earlier.

## Running from the command line


### Preparing the environment

- In your command prompt/ terminal, navigate to the main PerovStats folder (the folder with subfolders such as `/docs/` and `/src/` in it)

- Ensure you have started you virtual environment. If you do not see `(venv)` on the left of the command lines type:

    - **Windows:** `venv\Scripts\activate`
    - **macOS/Linux:** `source venv/bin/activate`

    `(venv)` should now appear and you are ready to run PerovStats

![(venv) showing in terminal](./images/venv_check.jpg)

### Running the program with default settings

- The program can be run with the command `perovstats`. This will use the default settings and take config options from `default_config.yaml` found in `/src/perovstats/default_config.yaml`. Please do not move this file.

### Running the program with custom settings

#### Command line arguments

- Other than inputting a custom config file (see [config documentation](config.md)) common arguments can be passed from the terminal when running PerovStats.

- The arguments available are:

    - `--help` or `-h`
        - Display a menu showing each command detailed here.
            ```
            perovstats --help
            ```

    - `--config_file` or `-c`
        - Give a custom config file which will override the default settings (again, see [configuration documentation](config.md) for details).

            ```
            perovstats -c "C:/path/to/config.yaml" process
            ```

    - `--base_dir` or `-d`
        - Select the folder on your computer you would like to look for data files in.

            **Note:** This requires a *folder* rather than a data file

            ```
            perovstats -d "C:/path/to/data_files/" process
            ```

    - `--file_ext` or `-e`
        - Select the file extension of the scans you have (e.g. ".spm" or ".001")

            ```
            perovstats -e ".spm" process
            ```

    - `--channel` or `-n`
        - Select the channel to use from the AFM scan (e.g. "Height")

            ```
            perovstats -n "Height" process
            ```

    - `--output_dir` or `-o`
        - Directory to output resultant images + data to

            **Note:** An `/output/` folder will be created inside this directory and files will be saved inside this

            ```
            perovstats -o "C:/path/to/output/folder/" process
            ```

    - `--segmentation` or `-s`
        - Choose the segmentation method for finding grains in an image. Options are 'traditional' or 'cellpose' (see the [segmentation documentation](segmentation.md) for details on each of these).

            ```
            perovstats -s "traditional" process
            ```

- You can include multiple arguments in one command by adding a space between them:

    ```
    perovstats -s cellpose -e ".spm" process
    ```
