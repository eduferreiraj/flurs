class MetaRecommender:
    def initialize(self, recommender):
        self.recommender = recommender
        self.recommender.register_observer(self)
        self._learn_rate = self.recommender.learn_rate


    def profile_difference(self, u_grad):
        return

    def new_user(self, u_id):
        return

    def parameters(self):
        return ""

    def update_model(self, ua, ia, rating):
        return
    def learn_rate(self, user):
        return self._learn_rate
