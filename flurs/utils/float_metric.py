class FloatMetric:
    def __init__(self, w_size):
        self.w_size = w_size
        self.floating_value = 0.0
        self.counter = 0
    def next(self, value):
        if self.counter < self.w_size:
            self.counter += 1
        self.update(value)
    def get(self):
        return self.floating_value
    def __repr__(self):
        return str(self.get())
    def update(self, value):
        return

class FloatMean(FloatMetric):
    def update(self, value):
        self.floating_value = (self.floating_value * (self.counter - 1) + value) / self.counter

class FloatSTD(FloatMetric):
    def __init__(self, mean):
        self.mean = mean
        super().__init__(self.mean.w_size)
    def update(self, value):
        self.floating_value = (self.floating_value * (self.counter - 1) + (value - self.mean.get())**2) / self.counter
    def get(self):
        return self.floating_value**.5
