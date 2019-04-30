import numpy as np
from ..baseforgetting import BaseForgetting
from numba import jit

class ForgetUnpopularItems(BaseForgetting):
    def __init__(self, alpha = 1.01):
        self.item_ratings = np.zeros(0)
        self.alpha = alpha
        self.coefs = []

    def reset_forgetting(self):
        self.item_ratings = np.zeros(0)

    def register_item(self, item):
        size_items = len(self.item_ratings)
        if size_items > item.index:
            return
        elif size_items == 0:
            self.item_ratings = np.zeros((item.index + 1, 1))
        else:
            diff = item.index - (size_items - 1)
            newMatrix = np.zeros((diff + 1, 1))
            self.item_ratings = np.concatenate((self.item_ratings, newMatrix))

    def update(self, user, item, rating):
        self.item_ratings[item] += 1
        return

    def item_forgetting(self, item_vec, item, last_item_vec):
        coef = -(self.alpha ** -self.item_ratings[item]) + 1
        next_i_vec = item_vec * coef
        self.coefs.append(coef)
        return next_i_vec

    def mean(self):
        return print("Alpha:{0} Mean:{1:2f}".format(self.alpha, np.mean(self.coefs)))
