from collections import defaultdict
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


def create_random_subsets(arr: Union[int, Iterable], max_num_subsets: int, max_retries: int = 10,
                          subset_length_mode: Union[str, int] = 'multi', max_subset_length: int = None,
                          replace: bool = False, seed: int = None, probs: list = None):
    """
    Create random subsets from an array.
    Each subset has a random length from 1 to (N-1), so it's never the whole input array.
    What array elements have been picked will have less probability of being picked in the next subset (all elements
    will be picked with roughly equal probability.), unless "probs" is specified.

    Args:
        arr: the list to select subsets from. If this is an int, the list will be list(range(n))
        max_num_subsets: number of subsets to create; return fewer subsets if cannot find enough unique subsets
        max_retries: maximum number of retries when encountering duplicate subsets
        subset_length_mode: accepted values are
            an integer: a given length for all subsets
            'multi': randomly generate a length for each subset
            'single': randomly generate the same length for all subsets
        max_subset_length: maximum length of a subset
        replace: whether to allow duplicate subsets in the result;
            this guarantees the function always returns ``max_num_subsets`` subsets
        seed: random seed
        probs: fixed probabilities for items in array to be picked in all subsets

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

    rand_generator = np.random.default_rng(seed)

    # initialise the number of elements to pick for each subset
    if max_subset_length is None:
        max_subset_length = n
    else:
        assert 1 <= max_subset_length < n, f'1 <= `max_subset_length` < len(`arr`); but found {max_subset_length}'
        max_subset_length = max_subset_length + 1

    if subset_length_mode == 'multi':
        len_subsets = rand_generator.integers(low=1, high=max_subset_length, size=max_num_subsets)
    elif subset_length_mode == 'single':
        len_subsets = [rand_generator.integers(low=1, high=max_subset_length)] * max_num_subsets
    elif isinstance(subset_length_mode, int):
        assert 1 <= subset_length_mode < n, f'1 <= `subset_length_mode` < len(`arr`); but found {subset_length_mode}'
        len_subsets = [subset_length_mode] * max_num_subsets
    else:
        raise ValueError(f'invalid input type for `subset_length_mode`: {subset_length_mode}')

    # initialise the probabilities for each unit
    if probs is None:
        p_units = np.ones(n, dtype=float)
    else:
        assert len(probs) == len(arr), \
            f'probs must have same length as input "arr", but found {len(probs)} and {len(arr)}'
        p_units = np.array(probs)
        p_units = p_units / p_units.sum()
    results_dict = defaultdict(int)  # key: subset as tuple; value: num appearances

    i = 0
    j = 0
    # for each subset
    while (i < len(len_subsets)) and (j < len(len_subsets) + max_retries):
        if probs is None:
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
        if step_result not in results_dict:
            results_dict[step_result] += 1
            i += 1

            # update unit probabilities
            if probs is None:
                p_units[list(step_result)] /= 2
        j += 1

    # convert output format
    results_list = []
    for subset, repeat in results_dict.items():
        results_list += [list(subset)] * repeat

    if isinstance(arr, np.ndarray):
        results_list = [arr[v].tolist() for v in results_list]

    # return duplicates if needed and `replace=True`
    if (len(results_list) < max_num_subsets) and replace:
        results_list += results_list[len(results_list) - max_num_subsets:].copy()

    return results_list
