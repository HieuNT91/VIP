import time
import functools 

def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed_s = end - start
        print(f"[PROFILE] {func.__name__} took {elapsed_s:.3f} s")
        return result
    return wrapper