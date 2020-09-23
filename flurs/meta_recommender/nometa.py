from .meta_recommender import MetaRecommender

class NoMeta(MetaRecommender):
    """The default method. Nothing is done. Just return.

    """
    def __init__(self):
        return

    def initialize(self, recommender):
        return
