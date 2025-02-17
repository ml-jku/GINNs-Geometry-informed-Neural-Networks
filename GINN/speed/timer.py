from time import perf_counter

import torch
from contextlib import nullcontext

class Timer:
    _instance = None
    
    def __init__(self, print=False, accumulate=False, lock=None):
        """
        Wrapper class for the timing context manager.
        This is a singleton class, it only has one instance, and in our case initializing several is no problem.
        Here are other options on how to define a singleton class:
        https://stackoverflow.com/questions/6760685/what-is-the-best-way-of-implementing-singleton-in-python
        """
        # making it a singleton
        self.__class__.__new__ = lambda _: self
        Timer._instance = self
        # proceed with initialization
        self.do_print = print
        self.do_accumulate = accumulate
        self.logbook = {}
        if lock is None:
            print(f'WARNING: No lock provided for TimerHelper')
        self.lock = lock if lock else nullcontext()  # Lock for the shared dictionary; only if lock is provided
        self.do_record = self.do_print or self.do_accumulate
        self.start = perf_counter()

    @staticmethod
    def record(name, instance=None):
        # check if singleton instance exists
        if instance is not None:
            return RecordTime(name, instance)
        
        if Timer._instance is None:
            return nullcontext()
        return RecordTime(name, Timer._instance)
        
    # def record(self, name):
    #     if self.do_record: return RecordTime(name, self, self.lock)
    #     return nullcontext()
    
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
    def __init__(self, name, timer) -> None:
        self.name = name
        self.timer_helper = timer
        self.lock = timer.lock
    
    def __enter__(self):
        self.start = perf_counter()
        self.start_mem = torch.cuda.memory_reserved()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        with self.lock:  # Synchronize access to the logbook
            if self.timer_helper.do_print:
                self.readout = f'Time for {self.name}: {self.time:.3f}'
                self.end_mem = torch.cuda.memory_reserved()
                print(self.readout)
                print(f'Memory before: {self.start_mem/2**30:0.1f}, after: {self.end_mem/2**30:0.1f}, diff: {(self.end_mem - self.start_mem)/2**30:0.1f}')
            if self.timer_helper.do_accumulate:
                if self.name in self.timer_helper.logbook:
                    self.timer_helper.logbook[self.name] += self.time
                else:
                    self.timer_helper.logbook[self.name] = self.time


if __name__ == '__main__':
    config = {'timer_print': True, 'timer_accumulate': True}
    # initialize the TimerHelper once
    timer = Timer(config)
    # use it if its defined
    with Timer.record('test'):
        print('Hello')
    with Timer.record('test2'):
        print('World')
        
    timer.print_logbook()