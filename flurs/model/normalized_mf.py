from sklearn.base import BaseEstimator
import numpy as np

class NormalizedMF(BaseEstimator):

    """Incremental Matrix Factorization

    J. Vinagre et al.
    "Fast Incremental Matrix Factorization for Recommendation with Positive-Only Feedback"
    In Proceedings of UMAP 2014, pages 459-470, July 2014.

    """

    def __init__(self, k, l2_reg, learn_rate, forgetting, norm_value = 1.):
        self.k = k
        self.learn_rate = learn_rate
        self.forgetting = forgetting
        self.norm_value = norm_value
        self.l2_reg = l2_reg
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

        i_grad = err * u_vec
        next_i_vec = i_vec + self.learn_rate * i_grad

        u_grad = err * i_vec
        next_u_vec = u_vec + lrate * u_grad

        self.forgetting.update(ua, ia, rating)
        next_i_vec = self.forgetting.item_forgetting(next_i_vec, ia, i_vec)
        next_u_vec = self.forgetting.user_forgetting(next_u_vec, ua, u_vec)


        self.A[ua] = next_u_vec
        self.B[ia] = next_i_vec

        if self.observer:
            self.observer.profile_difference(ua, u_grad)
            self.observer.update_model(ua, ia, rating)
	
        self.A[ua] = self.A[ua] * np.sqrt(self.norm_value / np.dot(self.A[ua], self.A[ua].T))
        self.B[ia] = self.B[ia] * np.sqrt(self.norm_value / np.dot(self.B[ia], self.B[ia].T))
	
