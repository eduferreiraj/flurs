class FloatMetric:
    def __init__(self, decay):
        self.decay = decay
        self.floating_value = 0.0

    def next(self, value):
        self.update(value)

    def get(self):
        return self.floating_value

    def __repr__(self):
        return str(self.get())

    def update(self, value):
        return

class FloatMean(FloatMetric):
    def update(self, value):
        self.floating_value = self.floating_value * self.decay + value * (1 - self.decay)

class FloatSTD(FloatMetric):
    def __init__(self, mean):
        self.mean = mean
        super().__init__(self.mean.decay)

    def update(self, value):
        self.floating_value = (self.floating_value * self.decay + (value - self.mean.get())**2) * (1 - self.decay)

    def get(self):
        return self.floating_value**.5

    def __repr__(self):
        return "{0:.10f}".format(self.get())
