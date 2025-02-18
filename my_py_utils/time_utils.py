import time
from datetime import datetime, timezone


def datetime_2_timestamp(dt: datetime, tz: int) -> int:
    """
    Convert a datetime object to timestamp

    Args:
        dt: datetime object
        tz: timezone of the datetime object

    Returns:
        timestamp in millisecond
    """
    dt = dt.replace(tzinfo=timezone.utc)
    timestamp = dt.timestamp() - tz * 3600
    return round(timestamp * 1000)


def str_2_timestamp(str_time: str, tz: int, str_format: str = '%Y/%m/%d %H:%M:%S') -> int:
    """
    Convert datetime as string to timestamp (msec)

    Args:
        str_time: string of datetime value
        tz: timezone of string value
        str_format: datetime string format, default is 'yyyy/mm/dd HH:MM:SS'

    Returns:
        timestamp in millisecond
    """
    dt = datetime.strptime(str_time, str_format)
    ts = datetime_2_timestamp(dt, tz)
    return ts


def timestamp_2_datetime(timestamp: int, tz: int) -> datetime:
    """
    Convert timestamp (millisecond) to a datetime object

    Args:
        timestamp: timestamp in millisecond
        tz: timezone to convert

    Returns:
        a datetime object
    """
    return datetime.utcfromtimestamp(timestamp / 1000 + tz * 3600)


def timestamp_2_str(timestamp: int, tz: int, str_format: str = '%Y/%m/%d %H:%M:%S') -> str:
    """
    Convert timestamp (msec) to string datetime

    Args:
        timestamp: timestamp in millisecond
        tz: timezone of the output string datetime
        str_format: format of the output string, default is 'yyyy/mm/dd HH:MM:SS'

    Returns:
        string datetime
    """
    dt = timestamp_2_datetime(timestamp, tz)
    str_time = dt.strftime(str_format)
    return str_time


class TimeThis:
    def __init__(self,
                 timer=time.time_ns,
                 printer=lambda x: print(f'Elapsed time: {x}'),
                 **kwargs):
        """
        Measure running time of a block of code

        Args:
            timer: function returning current time
            printer: print function; this class calls printer(duration, **kwargs) upon exit
            **kwargs: keyword args for the print function
        """
        self.timer = timer
        self.printer = printer
        self.printer_kwargs = kwargs

    def __enter__(self):
        self.start_time = self.timer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.printer(self.timer() - self.start_time, **self.printer_kwargs)
