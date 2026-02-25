from time import sleep, perf_counter

class Duration():
    def __init__(self, name):
        self.name = name
        self.prev = 0
        self.elapsed = 0
        self.fps = 0

    def set_prev(self):
        self.prev = perf_counter()

    def calc_elapsed(self):
        self.elapsed = perf_counter() - self.prev
        self.prev = perf_counter()
        self.calc_fps(self.elapsed)

    def calc_fps(self, elapsed):
        self.fps = 1/elapsed

    def print_fps(self):
        print(f'{self.name} fps : {self.fps:.1f}')

    def print_sec(self):
        print(f'{self.name} elapsed (sec) : {self.elapsed:.4f}')

    def get_elapsed(self):
        return self.elapsed
    
    def get_fps(self):
        return self.fps
