from ..base import RecommenderMixin
from ..model import BRISMF
from .. import logger

import numpy as np


class BRISMFRecommender(BRISMF, RecommenderMixin):

    """Biased Regularized Incremental Simultaneous Matrix Factorization

    References
    ----------
    - G. Takács et al.
        "Scalable collaborative filtering approaches for large recommender systems."
        J. Mach. Learn. Res. 10, 623–656 (2009)

    """

    def initialize(self, static=False):
        super(BRISMFRecommender, self).initialize()

        # if True, parameters will not be updated in evaluation
        self.static = static

    def register_user(self, user):
        super(BRISMFRecommender, self).register_user(user)
        u_vec = np.random.normal(0., 0.1, self.k)
        u_vec[0] = 1
        self.users[user.index]['vec'] = u_vec
    def register_item(self, item):
        super(BRISMFRecommender, self).register_item(item)
        i_vec = np.random.normal(0., 0.1, (1, self.k))
        i_vec[0][1] = 1
        if self.Q.size == 0:
            self.Q = i_vec
        else:
            self.Q = np.concatenate((self.Q, i_vec))

    def update(self, e, batch_train=False):
        # static baseline; w/o updating the model
        if not batch_train and self.static:
            return

        self.update_model(e.user.index, e.item.index, e.value)

    def score(self, user, candidates):
        pred = np.dot(self.users[user.index]['vec'],
                      self.Q[candidates, :].T)
        return np.abs(1. - pred.flatten())

    def recommend(self, user, candidates):
        scores = self.score(user, candidates)
        return self.scores2recos(scores, candidates)
