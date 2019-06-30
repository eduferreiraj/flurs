from .meta_recommender import MetaRecommender

class BUP(MetaRecommender):
    def __init__(self, recommender):
        super(MetaRecommender, self).__init__(recommender)
