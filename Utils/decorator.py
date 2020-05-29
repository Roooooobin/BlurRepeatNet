import datetime as dt
from functools import wraps


def time_trace(func):
    @wraps(func)
    def record_time(*args, **kwargs):
        st = dt.datetime.now()
        res = func(*args, **kwargs)
        ed = dt.datetime.now()
        time_cost = (ed-st).total_seconds()
        print("{} cost {}s".format(func.__name__, str(time_cost)))
        return res
    return record_time
