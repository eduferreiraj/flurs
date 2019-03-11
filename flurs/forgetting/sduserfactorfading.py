import numpy as np
from ..baseforgetting import BaseForgetting

class SDUserFactorFading(BaseForgetting):
    def __init__(self, alpha = 1.001):
        self.alpha = alpha

    def user_forgetting(self, user_vec, user, last_user_vec):
        diff = user_vec - last_user_vec
        stability = self.alpha ** (- np.std(diff**2))
        return user_vec * stability
