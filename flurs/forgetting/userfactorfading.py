import numpy as np
from ..baseforgetting import BaseForgetting

class UserFactorFading(BaseForgetting):
    def initialize(self, fade = .99):
        self.fade = fade

    def register_user(self):
        return

    def register_item(self):
        return

    def update(self, user, item, rating):
        return

    def item_forgetting(self, item_vec):
        return item_vec

    def user_forgetting(self, user_vec):
        return user_vec * fade
