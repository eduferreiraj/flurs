from .meta_recommender import MetaRecommender
from ..utils.float_metric import FloatSTD, FloatMean
import logging
class FloatLR(MetaRecommender):
    def __init__(self, lambda_l, lambda_s, alpha):
        self.lambda_s = lambda_s
        self.lambda_l = lambda_l
        self.alpha = alpha
        self.short_term = {}
        self.long_term = {}
        self.deviation = {}
        self.learn_vector = {}

        # create log configuration
        self.logger = logging.getLogger("experimenter.metarecommender")

    def register_user(self, u_id):
        if not u_id in self.short_term:
            self.short_term[u_id] = FloatMean(self.lambda_s)
            self.long_term[u_id] = FloatMean(self.lambda_l)
            self.deviation[u_id] = FloatSTD(self.long_term[u_id])
            self.learn_vector[u_id] = self.recommender.learn_rate

    def profile_difference(self, u_id, u_grad):
        self.update_metric(u_id, u_grad.std())
        if self.deviation[u_id].get() == 0.0:
            return
        diff = self.short_term[u_id].get() - self.long_term[u_id].get()
        change = (diff/self.deviation[u_id].get())
        change_coef = self.alpha**change
        self.learn_vector[u_id] *= change_coef

    def update_metric(self, u_id, value):
        self.short_term[u_id].next(value)
        self.long_term[u_id].next(value)
        self.deviation[u_id].next(value)

    def parameters_formater(self):
        return "Long Mean:{} Short Mean:{} Alpha:{}"

    def parameters(self):
        return [self.lambda_l, self.lambda_s, self.alpha]

    def learn_rate(self, user):
        return self.learn_vector[user]
