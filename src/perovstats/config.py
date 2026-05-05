from argparse import Namespace

def update_module(
    args: Namespace,
    perovstats_modules: tuple = (
        "process"
    ),
) -> None:
    """
    This function allows the sub-parser command to map to the pipeline we wish to use.

    >>> perovstats process

    Parameters
    ----------
    args : Namespace
        Default arguments that need parsing and updating.
    topostats_modules : tuple
        List of module names that are unique to TopoStats.
    """
    if args.module in perovstats_modules:
        args.module = "perovstats"
