from .meta_recommender import MetaRecommender
from ..utils.float_metric import FloatSTD, FloatMean
import logging
class UserAdaDelta(MetaRecommender):
    def __init__(self, decay, learn_rate=1.0, epsilon=0.0000001):
        self.decay = decay
        self.base_learn_rate = learn_rate
        self.user_mean = {}
        self.epsilon = epsilon
        # create log configuration
        self.logger = logging.getLogger("experimenter.metarecommender")

    def register(self, id):
        if not id in self.user_mean:
            self.user_mean[id] = FloatMean(self.decay)

    def profile_difference(self, _, id, grad):
        self.user_mean[id].update(grad.std())

    def learn_rate(self, id):
        return self.base_learn_rate / (self.learn_vector[id].get() + self.epsilon)
