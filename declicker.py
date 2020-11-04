import numpy as np
from math import log2
from scipy.interpolate import interp1d

from tqdm import trange
from itertools import chain

from matplotlib import pyplot as plt

from nptyping import NDArray
from break_points import BreakBands


def stretch_signal(signal: NDArray, target_lenght: int) -> NDArray:
    func = interp1d(np.linspace(0, 1, len(signal)), signal, kind='cubic')
    return func(np.linspace(0, 1, target_lenght))


def append_signal_with_xfade(whole: NDArray, part: NDArray, part_start: int, xfade_length: int) -> None:
    """Modifies 'whole' inplace!"""
    def xfade_power(s1, s2):
        # r = E[s1*s2] / sigma^2, where sigma = E[s1^2] = E[s2^2]
        r = np.mean(s1 * s2) / np.mean(np.concatenate((s1, s2)) ** 2)
        return (1 + log2(1 + r)) / 2

    k = xfade_power(whole[part_start:part_start+xfade_length], part[:xfade_length])
    xfade_multiplier = np.power(np.linspace(0, 1, xfade_length), k)  # fade-in

    whole[part_start:part_start+xfade_length] = (
        whole[part_start:part_start+xfade_length] * xfade_multiplier[::-1]
        + part[:xfade_length] * xfade_multiplier
    )
    whole[part_start+xfade_length:part_start+len(part)] = part[xfade_length:]


def test_xfade():
    out = np.array([0, 1, 2, 3, 2, 1, 0, -1, -2, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float')
    part = np.array([0, 0, 2, 3, 0, 0, -2, -3, 0, 0, 2, 3, 0, -2, -3], dtype='float')
    part_start = 4

    plt.plot(out, 'o-')
    plt.plot(np.arange(part_start, part_start+len(part)), part, 'o-')
    append_signal_with_xfade(out, part, part_start, 7)
    plt.plot(out, '.-')

    plt.show()


def declick_signal(signal: NDArray, break_bands: BreakBands, stretch_overlap: int = 1000) -> NDArray:
    out = np.zeros_like(signal)

    stretch_overlap = max(stretch_overlap, *[band[1]-band[0] for band in break_bands])  # samples

    interval_edges = [0] + list(chain.from_iterable(break_bands)) + [len(out)]
    intervals = list(zip(interval_edges[0::2], interval_edges[1::2]))

    target_itervals = [(max(start-stretch_overlap, 0), min(end+stretch_overlap, len(out))) for start, end in intervals]

    out[target_itervals[0][0]:target_itervals[0][1]] = stretch_signal(
        signal[intervals[0][0]:intervals[0][1]],
        target_itervals[0][1] - target_itervals[0][0]
    )
    for i in trange(1, len(target_itervals)):
        interval = intervals[i]
        target_interval = target_itervals[i]
        append_signal_with_xfade(
            whole=out,
            part=stretch_signal(signal[interval[0]:interval[1]], target_interval[1]-target_interval[0]),
            part_start=target_interval[0],
            xfade_length=target_itervals[i-1][1] - target_interval[0]
        )
    return out
