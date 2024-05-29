from time import perf_counter
from utils import set_and_true
from contextlib import nullcontext
from multiprocessing import Lock

class TimerHelper:
    def __init__(self, config, lock):
        """
        Wrapper class for the timing context manager.
        """
        self.config = config
        self.logbook = {}
        self.lock = lock  # Lock for the shared dictionary
        self.do_print = set_and_true('timer_print', config)
        self.do_accumulate = set_and_true('timer_accumulate', config)
        self.do_record = self.do_print or self.do_accumulate
        self.start = perf_counter()

    def record(self, name):
        if self.do_record: return RecordTime(name, self, self.lock)
        return nullcontext()
    
    def print_logbook(self, sort=True):
        """Print the accumulated timings recorded in the logbook."""
        with self.lock:
            print("Accumulated timings as ratio of total time and in s:")
            if sort: self.logbook = dict(sorted(dict(self.logbook).items(), key=lambda x:x[1], reverse=True))
            total_time = perf_counter() - self.start
            print(f'100.00% \t {total_time:03.1f} \t total time')
            for key, value in self.logbook.items():
                print(f'{value/total_time*100:04.2f}% \t {value:03.1f} \t {key}')


class RecordTime:
    def __init__(self, name, timer_helper, lock) -> None:
        self.name = name
        self.timer_helper = timer_helper
        self.lock = lock
    
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        with self.lock:  # Synchronize access to the logbook
            if self.timer_helper.do_print:
                self.readout = f'Time for {self.name}: {self.time:.3f}'
                print(self.readout)
            if self.timer_helper.do_accumulate:
                if self.name in self.timer_helper.logbook:
                    self.timer_helper.logbook[self.name] += self.time
                else:
                    self.timer_helper.logbook[self.name] = self.time
