from collections.abc import Iterable
from typing import List, Union
import bezier
import numpy as np
from scipy.interpolate import CubicSpline


def interp_resample(arr: np.ndarray, old_freq: float, new_freq: float) -> np.ndarray:
    """
    Resample by linear interpolation for every channel in the data array.

    Args:
        arr: data array, shape [length, channel]
        old_freq: old sampling rate
        new_freq: new sampling rate

    Returns:
        new data array, shape [new length, channel]
    """
    assert len(arr.shape) == 2, 'Only support 2D array'

    data_duration = len(arr) / old_freq
    new_arr = []
    for i in range(arr.shape[1]):
        old_ts = np.arange(0, data_duration, 1 / old_freq)
        new_ts = np.arange(0, data_duration, 1 / new_freq)
        new_channel = np.interp(x=new_ts, xp=old_ts, fp=arr[:, i])
        new_arr.append(new_channel)
    new_arr = np.stack(new_arr, axis=-1)
    return new_arr


def np_mode(array: np.ndarray, exclude_nan: bool = True) -> any:
    """
    Find mode value in a 1D array

    Args:
        array: an array of any shape, but it will be treated as a 1D array
        exclude_nan: whether to exclude nan values when finding mode

    Returns:
        the mode value, if `exclude_nan` is True and the whole input array is NaN, return None
    """
    if exclude_nan:
        array = array[~np.isnan(array)]
        if len(array) == 0:
            return None
    val, count = np.unique(array, return_counts=True)
    mode_ = val[np.argmax(count)]
    return mode_


def gen_random_curves(length: int, num_curves: int, sigma=0.2, knot=4, method: str = 'bezier',
                      randomizer: np.random._generator.Generator = None):
    """
    Generate random curves

    Args:
        length: length of the curve(s) to be generated
        num_curves: number of curves to be generated
        sigma: warping magnitude (std)
        knot: number of turns in the curve(s)
        method: method to connect random points to form a random curve;
            currently support [cubic|beizer]; default: bezier
        randomizer: numpy random generator (with seed)

    Returns:
        array shape [length, num curves]
    """
    xx = np.arange(0, length, (length - 1) / (knot + 1))
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, num_curves)) if randomizer is None \
        else randomizer.normal(loc=1.0, scale=sigma, size=(knot + 2, num_curves))

    if method == 'bezier':
        x_range = np.linspace(0, 1, num=length, endpoint=True)
        curves = [bezier.Curve.from_nodes(np.array([xx, yy[:, i]])).evaluate_multi(x_range)[1]
                  for i in range(num_curves)]
    elif method == 'cubic':
        x_range = np.arange(length)
        curves = [CubicSpline(xx, yy[:, i])(x_range) for i in range(num_curves)]
    else:
        raise ValueError(f'Available methods: [cubic|beizer], but received: {method}')

    curves = np.array(curves).T
    return curves


def interval_intersection(intervals: List[List[List[int]]]) -> List[List[int]]:
    """
    Find the intersection of multiple interrupted timeseries, each of which contains multiple segments represented by a
    pair of start & end timestamp.

    Args:
        intervals: a 3-level list;
            1st level: each element represents a timeseries;
            2nd level: each element is a segment in a timeseries;
            3rd level: 2 elements are start & end timestamps of a segment

    Returns:
        The intersection is also a timeseries with the same format as one element in the input list.
    """
    if len(intervals) == 0:
        raise ValueError("The input list doesn't have any element.")
    if len(intervals) == 1:
        return intervals[0]

    # index indicating current position for each interval list
    pos_indices = np.zeros(len(intervals), dtype=int)
    # lengths of all ranges
    all_len = np.array([len(interval) for interval in intervals])

    result = []
    while np.all(pos_indices < all_len):
        # the startpoint of the intersection
        lo = max([intervals[interval_idx][pos_idx][0] for interval_idx, pos_idx in enumerate(pos_indices)])
        # the endpoint of the intersection
        endpoints = [intervals[interval_idx][pos_idx][1] for interval_idx, pos_idx in enumerate(pos_indices)]
        hi = min(endpoints)

        # save result if there is an intersection among segments
        if lo < hi:
            result.append([lo, hi])

        # remove the interval with the smallest endpoint
        pos_indices[np.argmin(endpoints)] += 1

    return result


def create_random_unique_subsets(arr, max_num_subsets: int, max_retries: int = 10, seed: int = None):
    """
    Create random subsets from an array.
    All subsets are different from each other.
    Each subset has a random length from 1 to (N-1), so it's never the whole input array.
    What array elements have been picked will have less probability of being picked in the next subset. (all elements
    will be picked with roughly equal probability.)

    Args:
        arr: the list to select subsets from. If this is an int, the list will be list(range(n))
        max_num_subsets: number of subsets to create; return fewer subsets if cannot find enough unique subsets
        max_retries: maximum number of retries when encountering duplicate subsets
        seed: random seed

    Returns:
        a 2-level list, each child list is a subset
    """
    if isinstance(arr, Iterable):
        arr = np.array(arr)
        n = len(arr)
    elif isinstance(arr, int):
        n = arr
    else:
        raise ValueError(f'invalid input type for `arr`: {type(arr)}')

    # ensure there is at least one unit to pick from
    assert n >= 1, 'input list is empty.'

    # initialise the number of elements to pick for each subset
    rand_generator = np.random.default_rng(seed)
    len_subsets = rand_generator.integers(low=1, high=n, size=max_num_subsets)

    # initialise the probabilities for each unit
    p_units = np.ones(n, dtype=float)
    all_results = set()

    i = 0
    j = 0
    # for each subset
    while (i < len(len_subsets)) and (j < len(len_subsets) + max_retries):
        # normalise the probabilities, avoiding division by zero
        total_p_units = p_units.sum()
        if total_p_units == 0:
            p_units = np.ones(n, dtype=float) / n
        else:
            p_units /= total_p_units

        # select random elements based on the current probabilities
        step_result = rand_generator.choice(n, size=len_subsets[i], replace=False, p=p_units, shuffle=False)
        step_result = tuple(np.sort(step_result))

        # if result is unique, add it to all_results
        if step_result not in all_results:
            all_results.add(step_result)
            i += 1

            # update unit probabilities
            p_units[list(step_result)] /= 2
        j += 1

    all_results = [list(v) for v in all_results]

    if isinstance(arr, np.ndarray):
        all_results = [arr[v].tolist() for v in all_results]

    return all_results
