# PerovStats config documentation

## Using a custom config file

If you need to tweak the parameters defined in the [default config](../src/perovstats/default_config.yaml) you can create a new file to override the desired parameters.

A custom config file only needs to contain the parameters you want to change, any missing from this file will have their values taken from the default config.

Please ensure to indent your parameters correctly, including the names of each section

For example, do this:
```
segmentation:
    segmentation_method: traditional
```

Instead of this:
```
segmentation_method: traditional
```

### To create and use the config file:

- Open a text editor (just notepad will do if you don't have an IDE) and save the file in an accessible location called something like `config.yaml`.
- **Notebooks**:
    - Near the top of whichever notebook you are using is an option to define the config's filepath. Change the path in the double quotes to the path of this file you've just created.
- **Command line**:
    - When running PerovStats with `perovstats process` include `-c "path/to/config.yaml"` inbetween `perovstats` and `process` like so:

    ```
    perovstats -config_file "C:/Users/user1/Desktop/config.yaml" process
    ```
    - Alternatively you can just use `-c` in place of `-config_file`
