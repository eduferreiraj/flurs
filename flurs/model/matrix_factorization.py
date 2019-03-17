from sklearn.base import BaseEstimator
from ..forgetting import NoForgetting
import numpy as np
from numba import jit

class MatrixFactorization(BaseEstimator):

    """Incremental Matrix Factorization

    J. Vinagre et al.
    "Fast Incremental Matrix Factorization for Recommendation with Positive-Only Feedback"
    In Proceedings of UMAP 2014, pages 459-470, July 2014.

    """

    def __init__(self, k=40, l2_reg=.01, learn_rate=.003, forgetting=NoForgetting()):
        self.k = k
        self.l2_reg_u = l2_reg
        self.l2_reg_i = l2_reg
        self.learn_rate = learn_rate

        self.forgetting = forgetting
        self.forgetting.reset_forgetting()

        self.A = np.array([])
        self.B = np.array([])

    def register_item(self, item):
        super(BaseEstimator, self).register_item(item)
        self.forgetting.register_item(item)

    def register_user(self, user):
        super(BaseEstimator, self).register_user(user)
        self.forgetting.register_user(user)

    @jit
    def update_model(self, ua, ia, value):
        u_vec = self.A[ua]
        i_vec = self.B[ia]

        err = value - np.inner(u_vec, i_vec)

        grad = (err * i_vec - self.l2_reg_u * u_vec)
        next_u_vec = u_vec + self.learn_rate * grad

        grad = (err * u_vec - self.l2_reg_i * i_vec)
        next_i_vec = i_vec + self.learn_rate * grad

        self.forgetting.update(ua, ia, value)
        next_i_vec = self.forgetting.item_forgetting(next_i_vec, ia, i_vec)
        next_u_vec = self.forgetting.user_forgetting(next_u_vec, ua, u_vec)

        self.A[ua] = next_u_vec
        self.B[ia] = next_i_vec
