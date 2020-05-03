from .meta_recommender import MetaRecommender
import logging

class BUP(MetaRecommender):
    def __init__(self, boosted_lr, Detector):
        self.boosted_lr = boosted_lr
        self.Detector = Detector
        self.u_detectors = {}
        self.u_profile = {}
        # create log configuration
        self.logger = logging.getLogger("experimenter.metarecommender")

    def register(self, id):
        if id[0] == 'u':
            if not id in self.u_detectors:
                self.u_detectors[id] = self.Detector()

    def profile_difference(self, _, u_id, u_grad):
        if u_id[0] == 'u':
            self.u_detectors[u_id].add_element(u_grad.std())
            if self.u_detectors[u_id].detected_change():
                print("[{}] {}".format("C", u_id))
                if u_id in self.u_profile:
                    self.recommender.A[u_id] = self.u_profile[u_id]
                    del self.u_profile[u_id]
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
                next_u_vec = u_vec + self.boosted_lr * u_grad
                self.u_profile[u_id] = next_u_vec

    def parameters(self):
        return [self.boosted_lr, self.Detector.__name__, *self.detectors_param]
