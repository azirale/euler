from typing import Callable
import timeit

ProblemFunction = Callable[[], int]


def timed(base_func: ProblemFunction) -> ProblemFunction:
    def timed_func():
        s = timeit.default_timer()
        result = base_func()
        e = timeit.default_timer()
        print(f"Took {e-s:.3f}s")
        return result

    return timed_func

def time_me(func,*args,**kwargs):
    s = timeit.default_timer()
    x=func(*args,**kwargs)
    e = timeit.default_timer()
    print(f"{func.__name__} took {e-s:.3f}s")
    return x
