from ..utils.float_metric import FloatSTD, FloatMean

class FloatLR(MetaRecommender):
    def __init__(self, lambda_l, lambda_s, alpha):
        self.lambda_s = lambda_s
        self.lambda_l = lambda_l
        self.alpha = alpha
        short_term = {}
        long_term = {}
        deviation = {}

    def register_user(self, u_id):
        if not u_id in short_median:
            self.short_term[u_id] = FloatMean(lambda_s)
            self.long_term[u_id] = FloatMean(lambda_l)
            self.deviation[u_id] = FloatSTD(long_term[u_id])

    def profile_difference(self, u_id, diff):
        self.update_metric(u_id, diff.std())
        self.recommender.learn_rate *= self.alpha**((short_term[u_id] - long_term[u_id])/deviation[u_id])

    def update_metric(self, u_id, value):
        self.short_term[u_id].next(value)
        self.long_term[u_id].next(value)
        self.deviation[u_id].next(value)

    def parameters(self):
        return "(LW{},SW{},A{})".format(self.lambda_l, self.lambda_s, self.alpha)
