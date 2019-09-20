from sklearn.base import BaseEstimator
import numpy as np
from numba import jit

class BRISMF(BaseEstimator):

    """Biased Regularized Incremental Simultaneous Matrix Factorization

    References
    ----------
    - G. Takács et al.
        "Scalable collaborative filtering approaches for large recommender systems."
        J. Mach. Learn. Res. 10, 623–656 (2009)

    """

    def __init__(self, k, l2_reg, learn_rate, forgetting):
        self.k = k
        self.l2_reg_u = l2_reg
        self.l2_reg_i = l2_reg
        self.learn_rate = learn_rate
        self.forgetting = forgetting

        self.forgetting.reset_forgetting()

        self.A = np.array([])
        self.B = np.array([])
        self.observer = None

    def register_item(self, item):
        super(BaseEstimator, self).register_item(item)
        self.forgetting.register_item(item)

    def register_user(self, user):
        super(BaseEstimator, self).register_user(user)
        self.forgetting.register_user(user)


    def update_model(self, ua, ia, rating):
        """Update the model based in the paper and applying the forgetting technique.

        Args:
            ua (integer): User ID.
            ia (integer): Item ID.
            rating (integer): Rating.
        """

        if self.observer:
            self.observer.register_user(ua)
            lrate = self.observer.learn_rate(ua)
        else:
            lrate = self.learn_rate

        u_vec = self.A[ua]
        i_vec = self.B[ia]

        pred = np.inner(u_vec, i_vec)
        err = rating - pred

        u_grad = (err * i_vec - self.l2_reg_u * u_vec)
        next_u_vec = u_vec + lrate * u_grad

        i_grad = (err * u_vec - self.l2_reg_i * i_vec)
        next_i_vec = i_vec + self.learn_rate * i_grad

        next_u_vec[0] = 1.
        next_i_vec[1] = 1.

        self.forgetting.update(ua, ia, rating)
        next_i_vec = self.forgetting.item_forgetting(next_i_vec, ia, i_vec)
        next_u_vec = self.forgetting.user_forgetting(next_u_vec, ua, u_vec)


        self.A[ua] = next_u_vec
        self.B[ia] = next_i_vec

        if self.observer:
            u_diff = u_vec - next_u_vec
            self.observer.profile_difference(ua, u_grad)
