from ..base import RecommenderMixin
from ..model import BRISMF
from ..forgetting import NoForgetting
from numba import jit
import numpy as np


class BRISMFRecommender(BRISMF, RecommenderMixin):
    """Biased Regularized Incremental Simultaneous Matrix Factorization

    References
    ----------
    - G. Takács et al.
        "Scalable collaborative filtering approaches for large recommender systems."
        J. Mach. Learn. Res. 10, 623–656 (2009)

    """

    def initialize(self):
        """Initialize the BRISMF's model class.
        """
        super(BRISMF, self).initialize()

    def register_user(self, user):
        """Add matrix space to handle the new user if needed.

        Args:
            user (integer): User ID.

        """
        super(BRISMFRecommender, self).register_user(user)
        sizeA = len(self.A)
        if sizeA > user.index:
            return
        elif sizeA == 0:
            self.A = np.random.normal(0., 0.2, (user.index + 1, self.k))
        else:
            diff = user.index - (sizeA - 1)
            newMatrix = np.random.normal(0.,0.2,(diff, self.k))
            self.A = np.concatenate((self.A, newMatrix))
            self.logger.debug("Added {} lines to A. {}".format(diff+1, self.A.shape))
        self.A.T[0] = 1.

    def register_item(self, item):
        """Add matrix space to handle the new item if needed.

        Args:
            item (integer): Item ID.

        """
        super(BRISMFRecommender, self).register_item(item)
        sizeB = len(self.B)
        if sizeB > item.index:
            return
        elif sizeB == 0:
            self.B = np.random.normal(0., 0.2, (item.index + 1, self.k))
        else:
            diff = item.index - (sizeB - 1)
            newMatrix = np.random.normal(0.,0.2,(diff + 1, self.k))
            self.B = np.concatenate((self.B, newMatrix))
            self.logger.debug("Added {} lines to B. {}".format(diff+1, self.B.shape))
        self.B.T[1] = 1.

    def update(self, e):
        """Update in the model with the new event.

        Args:
            e (Event): New event.

        """
        self.update_model(e.user.index, e.item.index, e.rating)

    def score(self, user, candidates):
        """Multiply both user and all cadidates lines to get the user regression for each item.

        Args:
            user (integer): User ID.
            cadidates (numpy.Array): Integer vector with all candidates (item's id).
        Returns:
            numpy array: Vector with each score from the candidates.


        """
        pred = np.dot(self.A[user.index],
                      self.B[candidates].T)
        return pred.flatten()

    def recommend(self, user, candidates):
        """Get the score for each item and return the ordered vector and the score.

        Args:
            user (integer): User ID.
            cadidates (numpy.Array): Integer vector with all candidates.
        Returns:
            (numpy array, numpy array) : (Sorted list of items, Sorted scores).

        """
        scores = self.score(user, candidates)
        return self.scores2recos(scores, candidates)


    def reg_term(self, user_id, item_id):
        """Get the regularization term value from user_id and item_id to use in the SGD test.

        Args:
            user_id (integer): User ID.
            item_id (integer): Item ID.
        Returns:
            integer: Value from lambda times the sum of both norms squared

        """
        return self.l2_reg_u * (np.linalg.norm(self.A[user_id], 1)**2 + np.linalg.norm(self.B[item_id], 1)**2)
