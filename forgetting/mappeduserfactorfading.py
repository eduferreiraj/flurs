import numpy as np
from ..baseforgetting import BaseForgetting
from numba import jit

class MappedUserFactorFading(BaseForgetting):
    def __init__(self, alpha = 1.001):
        self.alpha = alpha
    def user_forgetting(self, user_vec, user, last_user_vec):
        diff = (user_vec - last_user_vec)
        stability = self.alpha ** (- np.sqrt(np.std(diff)))
        squared_diff = diff ** 2
        squared_diff = squared_diff / np.max(squared_diff)
        mappedStability = np.ones(user_vec.shape[0]) - (stability * squared_diff)
        return user_vec * mappedStability


    def parameters(self):
        return [self.alpha]
