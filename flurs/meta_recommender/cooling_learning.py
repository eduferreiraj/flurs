from .meta_recommender import MetaRecommender
import logging

class CoolingLearning(MetaRecommender):
    def __init__(self, N, delta, cool):
        self.N = N
        self.delta = delta
        self.users = {}
        self.cool = cool
        self.learn_rate = self.recommender.learn_rate
        self.logger = logging.getLogger("experimenter.metarecommender")

    def register_user(self, u_id):
        if not u_id in self.users:
            self.users[u_id] = 0

    def profile_difference(self, i_id, u_id, _):
        self.users[u_id] += 1

    def learn_rate(self, user):
        return self.learn_rate + self.learn_rate * self.delta * self.cool(self.users[user], self.N)
