from .meta_recommender import MetaRecommender
import logging

class BUP(MetaRecommender):
    def __init__(self, boosted_lr, Detector, *detectors_param):
        self.boosted_lr = boosted_lr
        self.Detector = Detector
        self.detectors_param = detectors_param
        self.u_detectors = {}
        self.u_profile = {}
        # create log configuration
        self.logger = logging.getLogger("experimenter.metarecommender")

    def register_user(self, u_id):
        if not u_id in self.u_detectors:
            self.u_detectors[u_id] = self.Detector(*self.detectors_param)

    def update_model(self, u_id, i_id, rating):
        if self.u_detectors[u_id].detected_change():
            print("[{}] {}".format("C", u_id))
            if u_id in self.u_profile:
                self.recommender.A[u_id] = self.u_profile[u_id]
                del self.u_profile
        elif self.u_detectors[u_id].detected_warning_zone():
            print("[{}] {}".format("W", u_id))
            i_vec = self.recommender.B[i_id]

            if not u_id in self.u_profile:
                u_vec = self.recommender.A[u_id]
            else:
                u_vec = self.u_profile[u_id]

            pred = np.inner(u_vec, i_vec)
            err = rating - pred

            u_grad = (err * i_vec - self.recommender.l2_reg_u * u_vec)
            next_u_vec = u_vec + lrate * u_grad

            self.u_detectors[u_id].add_element(u_grad.std())
        # else:
        #     print("[{}] {}".format("K", u_id))

    def profile_difference(self, u_id, u_grad):
        if not self.u_detectors[u_id].detected_warning_zone():
            self.u_detectors[u_id].add_element(u_grad.std())

    def parameters(self):
        return [self.boosted_lr, self.Detector.__class__.__name__, *self.detectors_param]
