# PerovStats segmentation

**Disclaimer:** Each method is still prone to inaccuracies and will not be perfect, but there are plans to improve them as time goes on.

## There are two segmentation methods available for you to use:

### Traditional

Traditional segmentation uses popular hard-coded segmentation methods to identify grain boundaries.

**Pros:**
- Relatively fast, approx 10 seconds for a 512x512 image
- Can still be sufficiently accurate if adjusted correctly for an image

**Cons:**
- Requires tweaking of parameters for different image sizes/ grain sizes
- Will contain some inaccuracies

### Cellpose ML

Cellpose is a machine learning software tool designed for complex grain finding for a range of images.

**Pros:**
- Much more generalisable than traditional segmentation, can adapt to new image/ grain sizes without manual tweaking
- Generally more accurate than traditional segmentation.


**Cons:**
- Takes significantly longer than traditional segmentation, approx 10mins per 512x512 image.
- Running for the first time requires the model to download, which is about 1GB and will take a few minutes to download. (This only needs to be downloaded once, it can be reused after this).

## Selecting the method

There are a number of ways to choose the segmentation method you want to use depending on how you use PerovStats. See above for details on each method and their benefits/ drawbacks.

### Notebooks
The third cell of the [batch processing notebook](../notebooks/perovstats-multi-process.ipynb) contains an editable `segmentation_method` variable. The options are "traditional" and "cellpose". Simply changing the text in this variable and re-running the notebook will switch the method it uses.

***Note:** The demo notebook runs both traditional and cellpose segmentation and displays the results side-by-side so you can see the difference, therefore method selection is not needed.*

### Command line
There are two ways to choose the method if you are running PerovStats from the command line:
- **Command line argument:**

    Using the `--segmentation` argument (alternatively just `-s`) followed by either `traditional` or `cellpose` will select this method for processing. The argument goes in between `perovstats` and `process` like so:

        perovstats -s cellpose process

- **Config file parameter:**

    If you're using a custom config file (see the [configuration documentation](config.md) for help with this) you can choose the segmentation method by adding this to the file:

        segmentation:
            segmentation_method: [traditional or cellpose]

    Using this custom config file will mean your selection here overrides the default option and this segmentation method will be used.
