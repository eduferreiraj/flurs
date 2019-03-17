import numpy as np
from ..baseforgetting import BaseForgetting
from numba import jit

class MappedUserFactorFading(BaseForgetting):
    def __init__(self, alpha = 1.001):
        self.alpha = alpha
    @jit
    def user_forgetting(self, user_vec, user, last_user_vec):
        squared_diff = (user_vec - last_user_vec) ** 2
        stability = self.alpha ** (- np.std(squared_diff))
        squared_diff = squared_diff / np.sum(squared_diff)
        mappedStability = stability * squared_diff
        return user_vec * mappedStability
