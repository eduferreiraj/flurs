import numpy as np

class BaseForgetting:
    def reset_forgetting(self):
        """Reset the model and get ready to use.
        """

    def register_user(self, user):
        """Add a new user to the model.

        Args:
            user (User): User object.
        """
        return

    def register_item(self, item):
        """Add a new item to the model.

        Args:
            item (Item): Item object.
        """
        return

    def update(self, user, item, rating):
        """Update the model using a new rating.

        Args:
            user (int): User index.
            item (int): Item index.
            rating (int): Rating given by user to the item.
        """
        return

    def item_forgetting(self, item_vec, item, last_item_vec):
        """Apply a forgetting operation in the item vector.

        Args:
            item_vec (numpy.array): Latent factor vector with item attributes updated.
            item (int): Item index.
            last_item_vec (numpy.array): Latent factor vector with item attributes not yet updated.
        """
        return item_vec

    def user_forgetting(self, user_vec, user, last_user_vec):
        """Apply a forgetting operation in the user vector.

        Args:
            user_vec (numpy.array): Latent factor vector with user attributes updated.
            user (int): User index.
            last_user_vec (numpy.array): Latent factor vector with user attributes not yet updated.
        """
        return user_vec

    def __repr__(self):
        if self.alpha:
            name = "{}(alpha={})".format(self.__class__.__name__, self.alpha)
        else:
            name = "{}()".format(self.__class__.__name__)
        return name

    def parameters(self):
        return ""

    def mean(self):
        return
