from ..baseforgetting import BaseForgetting

class NoForgetting(BaseForgetting):
    def __init__(self, alpha = None):
        self.alpha = alpha
        return
