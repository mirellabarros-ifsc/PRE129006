from collections.abc import Iterable
from math import erf

import numpy as np
import numpy.typing as npt


def is_inside(
    x: npt.NDArray[np.floating], center: float, width: float
) -> npt.NDArray[np.bool_]:
    a = center - width / 2
    b = center + width / 2
    return (a <= x) & (x < b)


def hist(
    data: npt.ArrayLike,
    bin_centers: Iterable[float],
) -> npt.NDArray[np.floating]:
    data = np.asarray(data)
    bin_centers = np.asarray(bin_centers)
    diffs = np.diff(bin_centers)
    if not np.allclose(diffs, diffs[0]):
        raise ValueError("`bin_centers` must be equally spaced")
    dx = diffs[0]
    edges = np.concatenate([bin_centers - dx / 2, [bin_centers[-1] + dx / 2]])
    hist, _ = np.histogram(data, bins=edges, density=True)
    return hist


def hist2(
    data1: npt.ArrayLike,
    data2: npt.ArrayLike,
    bin_centers1: Iterable[float],
    bin_centers2: Iterable[float],
) -> np.ndarray:
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    bin_centers1 = np.asarray(bin_centers1)
    bin_centers2 = np.asarray(bin_centers2)
    diffs1 = np.diff(bin_centers1)
    diffs2 = np.diff(bin_centers2)
    if not (np.allclose(diffs1, diffs1[0]) and np.allclose(diffs2, diffs2[0])):
        raise ValueError("bin_centers1 and bin_centers2 must be equally spaced")
    dx1, dx2 = diffs1[0], diffs2[0]
    edges1 = np.concatenate([bin_centers1 - dx1 / 2, [bin_centers1[-1] + dx1 / 2]])
    edges2 = np.concatenate([bin_centers2 - dx2 / 2, [bin_centers2[-1] + dx2 / 2]])
    hist, _, _ = np.histogram2d(data1, data2, bins=[edges1, edges2], density=True)
    return hist


@np.vectorize
def phi(x: float) -> float:
    return 0.5 * (1 + erf(x / np.sqrt(2)))


arrowprops = {
    "arrowstyle": "-|>",
    "color": "b",
    "linewidth": 3,
    "shrinkA": 0,
    "shrinkB": 0,
}
