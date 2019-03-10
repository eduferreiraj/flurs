import numpy as np
from ..baseforgetting import BaseForgetting

class ForgetUnpopularItems(BaseForgetting):
    def __init__(self, alpha = 1.01):
        self.items = np.zeros(0)
        self.alpha = alpha

    def reset_forgetting(self):
        self.items = np.zeros(0)


    def register_item(self, item):
        self.items = np.hstack((self.items, 0))

    def update(self, user, item, rating):
        self.items[item] += 1
        return

    def item_forgetting(self, item_vec, item):
        coef = -(self.alpha ** -self.items[item]) + 1
        next_i_vec = item_vec * coef
        return next_i_vec

    def user_forgetting(self, user_vec, user):
        return user_vec
