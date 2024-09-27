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

def print_dict_with_precision(data_dict: dict, decimal_points: int):
    """
    Prints the dictionary in one row with specified decimal points for float values.
    
    Args:
        data_dict (dict): Dictionary with string keys and float values.
        decimal_points (int): Number of decimal points to display for float values.
    """
    formatted_items = [f'{key}: {value:.{decimal_points}f}' for key, value in data_dict.items()]
    print(", ".join(formatted_items))

if __name__ == '__main__':
    print(rreplace(
        s='abcdefabcdefabcdef',
        old='a',
        new='A',
        occurrence=2
    ))
