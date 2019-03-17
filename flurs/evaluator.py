from .base import FeatureRecommenderMixin
from numba import jit
import time
import numpy as np

from collections import deque
from . import logger


class Evaluator(object):

    """Base class for experimentation of the incremental models with positive-only feedback.
    """

    def __init__(self, recommender, repeat=False, maxlen=None, debug=True):
        """Set/initialize parameters.

        Args:
            recommender (Recommender): Instance of a recommender which has been initialized.
            repeat (boolean): Choose whether the same item can be repeatedly interacted by the same user.
            maxlen (int): Size of an item buffer which stores most recently observed items.

        """
        self.rec = recommender
        self.feature_rec = issubclass(recommender.__class__, FeatureRecommenderMixin)

        self.repeat = repeat

        # create a ring buffer
        # save items which are observed in most recent `maxlen` events
        self.item_buffer = deque(maxlen=maxlen)

        self.debug = debug

    def fit(self, train_events, test_events, max_n_epoch=20):
        """Train a model using the first 30% positive events to avoid cold-start.

        Evaluation of this batch training is done by using the next 20% positive events.
        After the batch SGD training, the models are incrementally updated by using the 20% test events.

        Args:
            train_events (list of Event): Positive training events (0-30%).
            test_events (list of Event): Test events (30-50%).
            n_epoch (int): Number of epochs for the batch training.

        """
        # make initial status for batch training
        for e in train_events:
            self.__validate(e)
            e.user.known_item(e.item.index)
            self.item_buffer.append(e.item.index)

        # for batch evaluation, temporarily save new users info
        for e in test_events:
            self.__validate(e)
            self.item_buffer.append(e.item.index)

        self.__batch_update(train_events, test_events, max_n_epoch)

        # batch test events are considered as a new observations;
        # the model is incrementally updated based on them before the incremental evaluation step
        for e in test_events:
            e.user.known_item(e.item.index)
            self.rec.update(e)
    @jit
    def get_candidates(self, e):
        # check if the data allows users to interact the same items repeatedly
        unobserved = list(set(self.item_buffer))
        if not self.repeat:
            # make recommendation for all unobserved items
            every_item = np.arange(unobserved[-1] + 1)
            index = np.ones(every_item.shape[0], dtype=bool)
            index[e.user.known_items] = False
            unobserved = np.intersect1d(unobserved, every_item[index])
        np.random.shuffle(unobserved)
        unobserved = unobserved[:1000]
        candidates = np.append(unobserved, e.item.index)
        return candidates

    def evaluate(self, test_events):
        """Iterate recommend/update procedure and compute incremental recall.

        Args:
            test_events (list of Event): Positive test events.

        Returns:
            list of tuples: (rank, recommend time, update time)

        """
        for i, e in enumerate(test_events):
            self.__validate(e)

            candidates = self.get_candidates(e)
            # make top-{at} recommendation for the 1001 items
            start = time.clock()
            recos, scores = self.__recommend(e, candidates)
            recommend_time = (time.clock() - start)

            rank = np.where(recos == e.item.index)[0][0]

            # Step 2: update the model with the observed event
            e.user.known_item(e.item.index)
            start = time.clock()
            self.rec.update(e)
            update_time = (time.clock() - start)

            self.item_buffer.append(e.item.index)

            # (top-1 score, where the correct item is ranked, rec time, update time)
            yield scores[0], rank, recommend_time, update_time

    def __recommend(self, e, candidates):
        if self.feature_rec:
            return self.rec.recommend(e.user, candidates, e.context)
        else:
            return self.rec.recommend(e.user, candidates)

    def __validate(self, e):
        self.__validate_user(e)
        self.__validate_item(e)

    def __validate_user(self, e):
        if self.rec.is_new_user(e.user.index):
            self.rec.register_user(e.user)

    def __validate_item(self, e):
        if self.rec.is_new_item(e.item.index):
            self.rec.register_item(e.item)

    def __batch_update(self, train_events, test_events, max_n_epoch):
        """Batch update called by the fitting method.

        Args:
            train_events (list of Event): Positive training events.
            test_events (list of Event): Test events.
            n_epoch (int): Number of epochs for the batch training.

        """
        prev_err = np.inf
        curr_err = 10e20
        n_epoch = 0
        while curr_err/prev_err < 0.999999 and n_epoch <= max_n_epoch:
            n_epoch += 1
            np.random.shuffle(train_events)

            for e in train_events:
                self.rec.update(e)
            prev_err = curr_err
            curr_err = self.__batch_evaluate(test_events)
            if self.debug:
                logger.debug('epoch: %2d, conv: %f' % (n_epoch, curr_err/prev_err))

    def __batch_evaluate(self, test_events):
        """Evaluate the current model by using the given test events.

        Args:
            test_events (list of Event): Current model is evaluated by these events.

        Returns:
            float: Mean Percentile Rank for the test set.

        """
        sum_err = 0
        for i, e in enumerate(test_events):
            user = e.user
            item = e.item
            rating_prev = self.rec.score(user,[item.index])
            rating = e.value
            reg_term = self.rec.reg_term(user.index, item.index)
            sum_err += (rating - rating_prev) ** 2 + reg_term
        return sum_err
