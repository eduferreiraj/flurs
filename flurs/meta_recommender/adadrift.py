from .meta_recommender import MetaRecommender
from ..utils.float_metric import FloatSTD, FloatMean
import logging
class AdaDrift(MetaRecommender):
    def __init__(self, l_decay, s_decay, alpha):
        self.l_decay = l_decay
        self.s_decay = s_decay
        self.alpha = alpha
        self.short_term = {}
        self.long_term = {}
        self.deviation = {}
        self.learn_vector = {}

        # create log configuration
        self.logger = logging.getLogger("experimenter.metarecommender")

    def register(self, id):
        if not id in self.short_term:
            self.short_term[id] = FloatMean(self.s_decay)
            self.long_term[id] = FloatMean(self.l_decay)
            self.deviation[id] = FloatSTD(self.long_term[id])
            self.learn_vector[id] = self.recommender.learn_rate

    def profile_difference(self, _1,  _2, id, grad):
        self.update_metric(id, grad.std())
        if self.deviation[id].get() == 0.0:
            return
        diff = self.short_term[id].get() - self.long_term[id].get()
        change = (diff/self.deviation[id].get())
        change_coef = self.alpha**change
        self.learn_vector[id] *= change_coef

    def update_metric(self, id, value):
        self.short_term[id].next(value)
        self.long_term[id].next(value)
        self.deviation[id].next(value)

    def parameters_formater(self):
        return "Long Mean:{} Short Mean:{} Alpha:{}"

    def parameters(self):
        return [self.lambda_l, self.lambda_s, self.alpha]

    def learn_rate(self, id):
        return self.learn_vector[id]
