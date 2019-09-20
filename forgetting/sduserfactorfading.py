import numpy as np
from ..baseforgetting import BaseForgetting
from numba import jit

class SDUserFactorFading(BaseForgetting):
    def __init__(self, alpha = 1.001):
        self.alpha = alpha

    def user_forgetting(self, user_vec, user, last_user_vec):
        diff = user_vec - last_user_vec
        stability = self.alpha ** (- np.sqrt(np.std(diff)))
        return user_vec * stability

    def parameters(self):
        return [self.alpha]
