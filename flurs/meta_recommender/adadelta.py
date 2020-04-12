from .meta_recommender import MetaRecommender
from ..utils.float_metric import FloatSTD, FloatMean
import logging
class AdaDelta(MetaRecommender):
    def __init__(self, learn_rate=1.0, epsilon = 0.0000001,decay):
        self.learn_rate = learn_rate
        self.mean = FloatMean(decay)
        self.epsilon = epsilon

    def profile_difference(self, _, id, grad):
        self.mean.update(grad.std())
        self.variation = self.learn_rate / (self.mean.get() + self.epsilon)

    def learn_rate(self, id):
        return self.variation
