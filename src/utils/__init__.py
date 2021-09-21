import os
import time
import traceback


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def on_error(value, print_error_stack=True):
    """
    returns a wrapper which catches error within a function 
    and returns a default value on error
    value: the default value to be returned when error occured
    """
    def decorator(function):
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as e:
                if print_error_stack:
                    traceback.print_exc()
                return value
        return wrapper
    return decorator



def log_time(function):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = function(*args, **kwargs)
        end = time.time()
        print(
            "[{}] takes {:.5f} s"
            .format(function.__name__, end - start)
        )
        return res
    return wrapper
    