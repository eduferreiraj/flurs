class MetaRecommender:
    """ Meta Recommender

    Any meta recommender class should follow this minimum structure in
    order to allow interchangeability between all meta recommender
    methods.

    """

    def initialize(self, recommender):
        """Start the communication with the base recommender.

        Args:
            recommender (Recommeder): Recommender object.

        """
        self.recommender = recommender
        self.recommender.register_observer(self)
        self._learn_rate = self.recommender.learn_rate
        self.activated = False


    def profile_difference(self, id, u_grad):
        """Define the method header. This method receives the profile's id
        and the gradient used in the update.

        Args:
            recommender (Recommeder): Recommender object.

        """

        return

    def new_user(self, u_id):
        return

    def parameters(self):
        return ""

    def update_model(self, ua, ia, rating):
        return

    def learn_rate(self, user):
        return self._learn_rate

    def activate(self):
        self.activated = True
