import numpy as np
from ..baseforgetting import BaseForgetting

class UserFactorFading(BaseForgetting):
    def __init__(self, fade = 0.999999):
        self.fade = fade

    def user_forgetting(self, user_vec, user, last_user_vec):
        return user_vec * self.fade
