from ..base import RecommenderMixin
from ..model import MatrixFactorization
from .. import logger
from numba import jit

import numpy as np


class MFRecommender(MatrixFactorization, RecommenderMixin):

    """Incremental Matrix Factorization (MF) recommender

    References
    ----------

    - J. Vinagre et al.
      `Fast Incremental Matrix Factorization for Recommendation with Positive-only Feedback <http://link.springer.com/chapter/10.1007/978-3-319-08786-3_41>`_.
      In *Proc. of UMAP 2014*, pp. 459-470, July 2014.
    """

    def initialize(self, static=False):
        super(MFRecommender, self).initialize()

        # if True, parameters will not be updated in evaluation
        self.static = static

    def register_user(self, user):
        super(MFRecommender, self).register_user(user)
        sizeA = len(self.A)
        if sizeA >= user.index:
            return
        elif sizeA == 0:
            self.A = np.random.normal(0., 0.1, (user.index + 1, self.k))
        else:
            diff = user.index - sizeA
            newMatrix = np.random.normal(0.,0.1,(diff, self.k))
            self.A = np.concatenate((self.A, newMatrix))


    def register_item(self, item):
        super(MFRecommender, self).register_item(item)
        sizeB = len(self.B)
        if sizeB >= item.index:
            return
        elif sizeB == 0:
            self.B = np.random.normal(0., 0.1, (item.index + 1, self.k))
        else:
            diff = item.index - sizeB
            newMatrix = np.random.normal(0.,0.1,(diff + 1, self.k))
            self.B = np.concatenate((self.B, newMatrix))


    def update(self, e):
        self.update_model(e.user.index, e.item.index, e.value)

    @jit
    def score(self, user, candidates):
        pred = np.dot(self.A[user.index],
                      self.B[candidates].T)
        # print("Predicted: {}".format(pred))
        return np.abs(1. - pred.flatten())

    def recommend(self, user, candidates):
        scores = self.score(user, candidates)
        # print("Scores: {}".format(scores))
        return self.scores2recos(scores, candidates)
