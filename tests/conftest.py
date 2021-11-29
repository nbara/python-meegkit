import pytest

import matplotlib.pyplot as plt


def pytest_addoption(parser):
    """Add command line option to pytest."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--noplots",
        action="store_true",
        default=False,
        help="halt on plots"
    )


def pytest_collection_modifyitems(config, items):
    """Do not skip slow test if option provided."""
    if config.getoption("--noplots"):
        plt.switch_backend('agg')

    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
