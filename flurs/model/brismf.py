from sklearn.base import BaseEstimator
from .forgetting import NoForgetting

import numpy as np


class BRISMF(BaseEstimator):

    """Biased Regularized Incremental Simultaneous Matrix Factorization

    References
    ----------
    - G. Takács et al.
        "Scalable collaborative filtering approaches for large recommender systems."
        J. Mach. Learn. Res. 10, 623–656 (2009)

    """

    def __init__(self, k=40, l2_reg=.01, learn_rate=.003, forgetting=NoForgetting):
        self.k = k
        self.l2_reg_u = l2_reg
        self.l2_reg_i = l2_reg
        self.learn_rate = learn_rate
        self.forgetting = forgetting

        self.Q = np.array([])

    def update_model(self, ua, ia, value):
        u_vec = self.users[ua]['vec']
        i_vec = self.Q[ia]

        err = value - np.inner(u_vec, i_vec)

        grad = -2. * (err * i_vec - self.l2_reg_u * u_vec)
        next_u_vec = u_vec - self.learn_rate * grad

        grad = -2. * (err * u_vec - self.l2_reg_i * i_vec)
        next_i_vec = i_vec - self.learn_rate * grad

        forgetting.update(ua, ia, value)
        next_i_vec = forgetting.item_forgetting(next_i_vec)
        next_u_vec = forgetting.user_forgetting(next_u_vec)

        next_u_vec[0] = 1
        next_i_vec[1] = 1


        self.users[ua]['vec'] = next_u_vec
        self.Q[ia] = next_i_vec
