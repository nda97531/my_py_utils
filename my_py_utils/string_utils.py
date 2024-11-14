def rreplace(s: str, old: str, new: str, occurrence: int = 1):
    """
    Replace last occurrence of a substring
    Args:
        s: the whole string
        old: old substring
        new: new substring
        occurrence: number of occurrences

    Returns:
        new string
    """
    li = s.rsplit(old, occurrence)
    return new.join(li)


def format_dict_of_floats(data_dict: dict, decimal_points: int = 4, except_keys: set = None):
    """
    Format a dictionary into a string with specified decimal points for float values.

    Args:
        data_dict (dict): Dictionary with string keys and float values.
        decimal_points (int): Number of decimal points to display for float values.
        except_keys (set): Set of keys that should be in Python default format.
    """
    except_keys = except_keys or {}  # Default to empty list if None

    formatted_items = []
    for key, value in data_dict.items():
        if key in except_keys:
            formatted_items.append(f'{key}: {value}')
        else:
            formatted_items.append(f'{key}: {value:.{decimal_points}f}')
    return ", ".join(formatted_items)
