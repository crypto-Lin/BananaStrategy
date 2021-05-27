import datetime as dt
import functools
import time


class Timer():

        def __init__(self):
                self.start_dt = None

        def start(self):
                self.start_dt = dt.datetime.now()

        def stop(self):
                end_dt = dt.datetime.now()
                print('Time taken: %s' % (end_dt - self.start_dt))


def metric(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kw):
                t0 = time.time()
                back = fn(*args, **kw)
                print('%s executed in %s s' % (fn.__name__, time.time() - t0))
                return back

        return wrapper

