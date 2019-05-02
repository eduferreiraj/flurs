from .base import FeatureRecommenderMixin
from numba import jit
import time
import numpy as np

from collections import deque
from . import logger


class Evaluator(object):

    """Base class for experimentation of the incremental models with positive-only feedback.
    """

    def __init__(self, recommender, repeat=False, maxlen=None, debug=False):
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
            max_n_epoch (int): Maximum number of epochs for the batch training.

        """
        # make initial status for batch training
        for e in train_events:
            self.__validate(e)

        # for batch evaluation, temporarily save new users info
        for e in test_events:
            self.__validate(e)

        self.__batch_update(train_events, test_events, max_n_epoch)

        # batch test events are considered as a new observations;
        # the model is incrementally updated based on them before the incremental evaluation step
        for e in test_events:
            self.rec.update(e)

    def get_candidates(self, e):
        """Get a list of 1000 unknown items to the user.

        Args:
            e (Event): Event with the user .

        Returns:
            list of candidates: Integer

        """
        # select all known items in the model
        unobserved = list(set(self.item_buffer))
        every_item = np.arange(self.rec.B.shape[0])

        # create a boolean vector
        index = np.ones(self.rec.B.shape[0], dtype=bool)

        # set user known items to false
        index[e.user.known_items] = False

        # intersect between users unknown items and all items in the model
        unobserved = np.intersect1d(unobserved, every_item[index])

        # shuffle and select 1000
        np.random.shuffle(unobserved)
        unobserved = unobserved[:1000]
        candidates = np.append(unobserved, e.item.index)
        return candidates

    def evaluate(self, test_events):
        """Iterate recommend/update procedure and compute incremental recall.

        Args:
            test_events (list of Event): Positive test events.

        Returns:
            list of tuples: (score recall@1, rank, recommend time, update time)

        """
        for i, e in enumerate(test_events):
            scores, rank, recommend_time = self.recommend_event(e)

            # Step 2: update the model with the observed event
            self.__validate(e)
            start = time.clock()
            self.rec.update(e)
            update_time = (time.clock() - start)

            # (top-1 score, where the correct item is ranked, rec time, update time)
            yield scores[0], rank, recommend_time, update_time, e.user.index

    def recommend(self, test_events):
        """Just recommend, without updating the model.

        Args:
            test_events (list of Event): Positive test events.

        Returns:
            list of tuples: (score recall{at}1, rank, recommend time)

        """
        for i, e in enumerate(test_events):
            scores, rank, recommend_time = self.recommend_event(e)
            yield scores[0], rank, recommend_time

    def recommend_event(self, e):
        self.__validate(e)

        candidates = self.get_candidates(e)
        # make top-{at} recommendation for the 1001 items
        start = time.clock()
        recos, scores = self.__recommend(e, candidates)
        recommend_time = (time.clock() - start)

        rank = np.where(recos == e.item.index)[0][0]
        return scores, rank, recommend_time


    def __recommend(self, e, candidates):
        if self.feature_rec:
            return self.rec.recommend(e.user, candidates, e.context)
        else:
            return self.rec.recommend(e.user, candidates)

    def __validate(self, e):
        e.user.known_item(e.item.index)
        self.item_buffer.append(e.item.index)
        self.rec.register_user(e.user)
        self.rec.register_item(e.item)

    def __batch_update(self, train_events, test_events, max_n_epoch):
        """Batch update called by the fitting method.

        Args:
            train_events (list of Event): Positive training events.
            test_events (list of Event): Test events.
            n_epoch (int): Number of epochs for the batch training.

        """
        prev_err = np.inf
        curr_err = 100
        n_epoch = 1
        n_chunks = 20
        np.random.shuffle(train_events)
        train_chunks = np.array_split(train_events, n_chunks)
        convergence = np.inf
        converged = False
        convergence_criteria = .00001
        while not converged and n_epoch <= max_n_epoch:
            np.random.shuffle(train_chunks)
            for chunk in train_chunks:
                for e in chunk:
                    self.rec.update(e)
                prev_err = curr_err
                curr_err = self.__batch_evaluate(test_events)
                convergence = curr_err/prev_err - 1
                # convergence rate between (- convergence_criteria, convergence_criteria)
                converged = convergence < convergence_criteria and convergence > - convergence_criteria
                if converged:
                    break
            if self.debug:
                logger.debug('epoch: %2d, conv: %f, err: %d' %
                    (n_epoch, convergence, curr_err))
            n_epoch += 1
        self.rec.forgetting.mean()
        logger.info('Epochs:{} Convergence:{}'.format(n_epoch, convergence))


    def __batch_evaluate(self, test_events):
        """Evaluate the current model by using the given test events.

        Args:
            test_events (list of Event): Current model is evaluated by these events.

        Returns:
            float: Mean Percentile Rank for the test set.

        """
        sum_err = 0
        for i, e in enumerate(test_events):
            self.__validate(e)
            user = e.user
            item = e.item
            rating_prev = self.rec.score(user,[item.index])[0]
            rating = e.value
            reg_term = self.rec.reg_term(user.index, item.index)
            sum_err += (rating - rating_prev) ** 2 + reg_term
        return sum_err
