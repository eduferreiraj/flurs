import numpy as np
from ..baseforgetting import BaseForgetting
from numba import jit

class SDUserFactorFading(BaseForgetting):
    def __init__(self, alpha = 1.001):
        self.alpha = alpha
        self.coefs = []
    @jit
    def user_forgetting(self, user_vec, user, last_user_vec):
        diff = user_vec - last_user_vec
        # print("Diferenca: {}".format(diff))
        stability = self.alpha ** (- np.std(diff))
        # print("Norma: {}, Coef: {}, Std: {}".format(np.linalg.norm(diff), stability, np.std(diff)))
        self.coefs.append(stability)
        return user_vec * stability

    def mean(self):
        return print("Alpha:{0} Mean:{1:2f}".format(self.alpha, np.mean(self.coefs)))
