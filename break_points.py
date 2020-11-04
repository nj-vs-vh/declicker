import numpy as np

from matplotlib import pyplot as plt

from typing import List, Tuple
from nptyping import NDArray


def calculate_and_plot_diffs(signal: NDArray):
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

    deltas = np.abs(np.diff(signal))
    ax1.hist(deltas, density=True, bins=1000, label=r'$\Delta_i \equiv s_i - s_{i-1}$ distribution')
    ax1.set_yscale('log')
    ax1.legend()

    log_deltas = np.log10(deltas)
    ax2.hist(
        log_deltas[np.logical_not(np.isinf(log_deltas))], density=True, bins=1000,
        label=r'$log_{10} \Delta$ distribution'
    )
    ax2.set_yscale('log')
    ax2.legend()

    ma_deltas = 0.5 * (deltas[0:-2] + deltas[2:])
    second_deltas = np.abs(deltas[1:-1] - ma_deltas)

    ax3.hist(
        second_deltas, density=True, bins=1000,
        label=r'$\left( \Delta_i - \frac{\Delta_{i-1} + \Delta_{i+1}}{2} \right)$ distribution'
    )
    ax3.set_yscale('log')
    ax3.legend()

    plt.show()

    return second_deltas  # used as break point indicator


BreakBands = List[Tuple[int, int]]


def get_break_bands(break_point_indicator: NDArray, threshold: float) -> BreakBands:
    break_points = np.argwhere(break_point_indicator > threshold)[:, 0] + 3

    bands = []
    merging_window = 20  # samples
    band_padding = 3  # samples
    break_points = np.sort(break_points)
    current_band_start, current_band_end = break_points[0], break_points[0]
    for point in break_points:
        if point - current_band_end < merging_window:
            current_band_end = point
        else:
            bands.append((current_band_start - band_padding, current_band_end + band_padding))
            current_band_start, current_band_end = point, point
    return bands


def calculate_and_plot_break_regions(
    signal: NDArray, break_point_indicator: NDArray, threshold: float, max_break_count: int = 35
) -> BreakBands:
    fragment_width = 50  # samples, plotting option

    break_bands = get_break_bands(break_point_indicator, threshold)
    break_count = len(break_bands)

    if break_count == 0:
        print("No breaks found, try lower threshold")
    elif break_count > max_break_count:
        print(f'Too much breaks ({break_count}), try higher threashold or max_break_count parameters')
    else:
        print(f'{break_count} breaks')
        _, axes = plt.subplots(1, break_count, figsize=(8 * break_count, 8))
        for i, break_band in enumerate(break_bands):
            break_center = int(0.5 * (break_band[0] + break_band[1]))
            start = (break_center - fragment_width)
            stop = (break_center + fragment_width + 1)
            axes[i].plot(range(start, stop), signal[start:stop], '.-')
            axes[i].axvline(x=break_band[0], color='r')
            axes[i].axvline(x=break_band[1], color='r')
        plt.show()
        return break_bands
    return None


def plot_break_positions(signal: NDArray, break_bands: BreakBands):
    fig = plt.figure(figsize=(8 * len(break_bands), 16))
    ax = fig.gca()

    plt.plot(signal)
    for break_band in break_bands:
        center = int(0.5 * (break_band[0] + break_band[1]))
        ax.axvline(x=center, color='r')

    plt.show()
