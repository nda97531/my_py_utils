import numpy as np
import pandas as pd
import polars as pl
from typing import Union


def cal_num_windows(len_data: int, window_size: int, step_size: int, rounding: str = None):
    """
    Calculate the number of windows that would be produced with `sliding_window`.
    Args:
        len_data: number of data points in the time series.
        window_size: window size.
        step_size: step size.
        rounding: 'floor', 'ceil', or None.

    Returns:
        number of windows.
    """
    num_windows = (len_data - window_size) / step_size + 1
    if rounding == 'floor':
        num_windows = int(num_windows)
    elif rounding == 'ceil':
        num_windows = np.ceil(num_windows).astype(int)
    elif rounding is not None:
        raise ValueError(f'rounding must be "floor" or "ceil" or None, found {rounding}')
    return num_windows


def find_window_idx(window_idx: int, window_size: int, step_size: int) -> tuple:
    """
    Find start (inclusive) and end (exclusive) indices of a window.

    Args:
        window_idx: index of window.
        window_size: window size.
        step_size: step size.

    Returns:
        a tuple of start (inclusive) and end (exclusive) indices.
    """
    start_idx = window_idx * step_size
    end_idx = start_idx + window_size
    return start_idx, end_idx


def get_one_window(data: Union[np.ndarray, pl.DataFrame, pd.DataFrame], window_idx: int, window_size: int,
                   step_size: int, get_last=False) -> np.ndarray:
    """
    Get one window from data array.

    Args:
        data: data of shape [time, channel].
        window_idx: window index.
        window_size: window size.
        step_size: step size.
        get_last: whether to take the last rows as an addition window

    Returns:
        one window of shape [window size, channel].
    """
    start_idx, end_idx = find_window_idx(window_idx, window_size, step_size)

    if get_last:
        if start_idx >= len(data):
            raise IndexError(f'Window index [{start_idx}:{end_idx}] out of range for data len {len(data)}')
        elif end_idx > len(data):
            end_idx = len(data)
            start_idx = end_idx - window_size
    elif end_idx > len(data):
        raise IndexError(f'Window index [{start_idx}:{end_idx}] out of range for data len {len(data)}')

    if isinstance(data, pd.DataFrame):
        window = data.iloc[start_idx:end_idx]
    else:
        window = data[start_idx:end_idx]
    return window


def sliding_window(data: np.ndarray, window_size: int, step_size: int, get_last: bool = False) -> np.ndarray:
    """
    Sliding window along the first axis of the input array.
    Args:
        data: array shape [data length, ...]
        window_size: window size (num rows)
        step_size: step size (num rows)
        get_last: whether to take the last rows as an additional window if they are not already included

    Returns:
        array shape [num window, window length, ...]
    """
    num_windows = cal_num_windows(len_data=len(data), window_size=window_size, step_size=step_size)
    if num_windows < 1:
        return np.empty([0, window_size, *data.shape[1:]], dtype=data.dtype)
    num_windows_floor = int(num_windows)

    # if possible, run fast sliding window
    if window_size % step_size == 0:
        result = np.empty([num_windows_floor, window_size, *data.shape[1:]], dtype=data.dtype)
        div = int(window_size / step_size)
        for window_idx, data_idx in enumerate(range(0, window_size, step_size)):
            new_window_data = data[data_idx:data_idx + (len(data) - data_idx) // window_size * window_size].reshape(
                [-1, window_size, *data.shape[1:]])

            new_window_idx = list(range(window_idx, num_windows_floor, div))
            result[new_window_idx] = new_window_data
    # otherwise, run a regular loop
    else:
        result = np.array([data[i:i + window_size] for i in range(0, len(data) - window_size + 1, step_size)])

    if get_last and (num_windows % 1 != 0):
        result = np.concatenate([result, [data[-window_size:]]])

    return result


def shifting_window(data: np.ndarray, window_size: int, max_num_windows: int, min_step_size: int,
                    start_idx: int, end_idx: int) -> np.ndarray:
    """
    Get window(s) from an array while ensuring a certain part of the array is included.

    Args:
        data: array shape [data length, ...]
        window_size: window size (num rows)
        max_num_windows: desired number of windows, the actual number returned maybe smaller,
            depending on `min_step_size`
        min_step_size: only get multiple windows if 2 windows are farther than `min_step_size` from each other
        start_idx: start index of the required part (inclusive)
        end_idx: end index of the required part (inclusive)

    Returns:
        array shape [num window, window size, ...]
    """
    # end index inclusive -> exclusive
    end_idx = end_idx + 1

    # exception 1: data array is too small compared to window size
    if len(data) < window_size:
        return np.empty([0, window_size, *data.shape[1:]], dtype=data.dtype)
    elif len(data) == window_size:
        return np.expand_dims(data, axis=0)

    # exception 2: required part is not entirely in data array
    end_idx = min(end_idx, len(data))
    start_idx = max(start_idx, 0)

    # data array is large enough to contain both required part and window size
    first_window_start_idx = max(min(end_idx - window_size, start_idx), 0)
    last_window_start_idx = min(max(end_idx - window_size, start_idx), len(data) - window_size)
    assert last_window_start_idx >= first_window_start_idx

    # if there's only 1 window
    if first_window_start_idx == last_window_start_idx:
        windows = np.expand_dims(data[first_window_start_idx:first_window_start_idx + window_size], axis=0)
        return windows

    # otherwise, get windows by linspace
    max_num_windows = min(max_num_windows,
                          np.ceil((last_window_start_idx - first_window_start_idx + 1) / min_step_size).astype(int))
    window_start_indices = np.linspace(first_window_start_idx, last_window_start_idx, num=max_num_windows,
                                       endpoint=True, dtype=int)

    windows = np.array([data[i:i + window_size] for i in window_start_indices])
    return windows


if __name__ == '__main__':
    test = shifting_window(
        data=np.arange(1000),
        window_size=10,
        max_num_windows=3,
        min_step_size=5,
        start_idx=1,
        end_idx=5
    )

    print(test.shape)
    print(test)
