class MetaRecommender:
    def initialize(self, recommender):
        self.recommender = recommender
        self.recommender.register_observer(self)


    def profile_difference(self, diff):
        return

    def new_user(self, u_id):
        return

    def parameters(self):
        return ""
