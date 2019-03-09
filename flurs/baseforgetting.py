import numpy as np

class BaseForgetting:
    def initialize(self):
        """Initialize the forgetting technique with the parameters.
        """

        self.rating_matrix = np.zeros([[]])
        self.n_items = 0
        self.n_users = 0

    def register_user(self):
        """Add a new user to the matrix.
        """
        self.rating_matrix = np.vstack((self.rating_matrix, np.zeros(self.n_items)))
        self.n_users += 1

    def register_item(self):
        """Add a new item to the matrix.
        """
        self.rating_matrix = np.hstack((self.rating_matrix, np.zeros(self.n_users)))
        self.n_items += 1

    def update(self, user, item, rating):
        """Apply a forgetting operation in the item vector.

        Args:
            user (int): User index.
            item (int): Item indexself.
            rating (int): Rating given by user to the item.
        """
        return

    def item_forgetting(self, item_vec, item):
        """Apply a forgetting operation in the item vector.

        Args:
            item_vec (numpy.array): Latent factor vector with item attributes updated.
            item (int): Item index.
        """
        return item_vec
    def user_forgetting(self, user_vec, user):
        """Apply a forgetting operation in the user vector.

        Args:
            user_vec (numpy.array): Latent factor vector with user attributes updated.
            user (int): User index.
        """
        return user_vec
