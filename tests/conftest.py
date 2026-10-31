from collections.abc import Iterator

import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture()
def existing_figures() -> Iterator[set[int]]:
    from matplotlib import pyplot as plt

    figures = set(plt.get_fignums())
    yield figures
    for number in set(plt.get_fignums()) - figures:
        plt.close(number)
