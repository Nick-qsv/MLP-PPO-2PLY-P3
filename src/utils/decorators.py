# utils/decorators.py

import time
from functools import wraps

profiling_data = {}


def profile(func):
    @wraps(func)
    def wrapper_profile(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        func_name = func.__name__
        if func_name not in profiling_data:
            profiling_data[func_name] = {"total_time": 0.0, "call_count": 0}
        profiling_data[func_name]["total_time"] += elapsed
        profiling_data[func_name]["call_count"] += 1
        return result

    return wrapper_profile
