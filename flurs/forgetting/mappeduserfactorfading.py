import numpy as np
from ..baseforgetting import BaseForgetting
from numba import jit

class MappedUserFactorFading(BaseForgetting):
    def __init__(self, alpha = 1.001):
        self.alpha = alpha
        self.coefs = []
    @jit
    def user_forgetting(self, user_vec, user, last_user_vec):
        diff = (user_vec - last_user_vec)
        stability = self.alpha ** (- np.std(diff))
        squared_diff = diff ** 2
        squared_diff = squared_diff / np.max(squared_diff)
        mappedStability = np.ones(user_vec.shape[0]) - (stability * squared_diff)
        # print("\nStability: {}\n [SqDiff, Mapped]: {}".format(stability, list(zip(squared_diff, mappedStability))))
        self.coefs.append(np.mean(mappedStability))
        return user_vec * mappedStability

    def mean(self):
        return print("Alpha:{0} Mean:{1:2f}".format(self.alpha, np.mean(self.coefs)))
