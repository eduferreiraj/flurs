from ..baseforgetting import BaseForgetting

class NoForgetting(BaseForgetting):
    def initialize(self):
        return

    def register_user(self):
        return

    def register_item(self):
        return

    def update(self, user, item, rating):
        return

    def item_forgetting(self, item_vec):
        return item_vec

    def user_forgetting(self, user_vec):
        return user_vec
