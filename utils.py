import time

def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} Time: {time.time() - start_time:.2f} second")
        return result
    return wrapper
