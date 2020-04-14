from .meta_recommender import MetaRecommender
from ..utils.float_metric import FloatSTD, FloatMean
import logging
class AdaDelta(MetaRecommender):
    def __init__(self, decay, learn_rate=1.0, epsilon = 1e-6):
        self.base_learn_rate = learn_rate
        self.mean = FloatMean(decay)
        self.epsilon = epsilon

    def profile_difference(self, _, id, grad):
        self.mean.update(grad.std())
        self.variation = self.base_learn_rate / (self.mean.get() + self.epsilon)

    def learn_rate(self, id):
        return self.variation

    def register(self, id):
        return
