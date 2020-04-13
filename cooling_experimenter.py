from flurs.recommender import BRISMFRecommender, MFRecommender
from flurs.evaluator import Evaluator
from flurs.meta_recommender import CoolingLearning
from os.path import exists
import logging, os, sys, datetime, smtplib, traceback, time
import numpy as np

class Recall:
    def __init__(self, n):
        self.n = n
        self.hits = 0
        self.stream = 0
        self.r_mean = 0.0
        self.recall_list = []
    def update(self, rank):
        if rank <= self.n:
            self.hits += 1
        self.stream += 1
        recall = self.score()
        self.recall_list.append(recall)
        return recall
    def score(self):
        return self.hits/self.stream
    def mean(self):
        return np.mean(self.recall_list[20000:])


class Configuration:
    def __init__(self, data_path, Recommender, k=80, l2_reg= .01, learn_rate=.02, forgetting=NoForgetting(alpha=None), meta=NoMeta(), seed=None, exp_name = ""):
        self.seed = seed
        self.train_size = .2
        self.test_size = .3
        self.status = "..."
        self.configuration = {
            "recommender" : Recommender.__name__,
            "learn_rate" : learn_rate,
            "k" : k,
            "forgetting": " - ",
            "forgetting_param": " - ",

            "meta" : " - ",
            "meta_param": " - "
        }

        self.exp_name = exp_name

    def recommender(self):
        return self.__recommender

    def get_status(self):
        return "\n[{}]\t{}".format(self.status, self.exp_name.replace("_", " "))

    def get_report(self):
        return """> Experiment Report
    - Dataset: {}.
    - Result File: {}.
    - Status: {}.

> Experiment Configuration
    Recommender
    - Name: {}
    - Learning Rate: {}
    - K Dimensions: {}

    Meta Recommender
    - Name: {}
    - Paramenters: {}

    Forgetting Techniques
    - Name: {}
    - Paramenters: {}
        """.format(self.data_path.split("/")[-1], "{}.dat".format(self.exp_name), self.status, self.configuration["recommender"], self.configuration["learn_rate"], self.configuration["k"], self.configuration["meta"], self.configuration["meta_param"], self.configuration["forgetting"], self.configuration["forgetting_param"])

    def get_batch_data(self):
        n_batch_train = int(self.data.n_sample * self.train_size)  # 20% for pre-training to avoid cold-start
        n_batch_test = int(self.data.n_sample * self.test_size)  # 30% for evaluation of pre-training
        batch_tail = n_batch_train + n_batch_test

        # pre-train
        # 20% for batch training
        batch_training = self.data.samples[:n_batch_train]
        # 30% for batch evaluate
        batch_test = self.data.samples[n_batch_train:batch_tail]
        # after the batch training, 30% samples are used for incremental updating

        return batch_training, batch_test

    def get_prequential_data(self):
        n_batch_train = int(self.data.n_sample * self.train_size)  # 20% for pre-training to avoid cold-start
        n_batch_test = int(self.data.n_sample * self.test_size)  # 30% for evaluation of pre-training
        batch_tail = n_batch_train + n_batch_test

        return self.data.samples[batch_tail:]

    def start(self):
        self.log_file = BASE_PATH + LOG_PATH + self.exp_name + "_" +  datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + ".log"
        self.logger = logging.getLogger("experimenter.jupyter")
#         self.file_handler = logging.FileHandler(self.log_file, mode='w+')
#         self.file_handler.setLevel(logging.DEBUG)
#         self.logger.handlers = [self.file_handler, console]

        self.start_time = time.process_time()
        self.logger.info('converting data into FluRS input object')

        self.result_file = BASE_PATH + "flurs/results/k{}/{}.dat".format(self.configuration["k"], self.exp_name)
        if exists(self.result_file):
            return
        self.data = csv_loader(self.data_path)
        self.logger.info('initialize recommendation model {}'.format(self.exp_name))
        self.__recommender.initialize()
        self.meta.initialize(self.__recommender)
        return self

    def finish(self):
        if self.start_time != None:
            self.finish_time = time.process_time()
            del self.__recommender
            del self.data
            self.duration =  (self.finish_time - self.start_time)/60
            self.status = "OK!"
#             self.file_handler.close()
        else:
            self.logger.warning("Experiment not finished properly. Please start it before finish")

class Experimenter:
    def __init__(self):
        self.configurations = []
        self.gmail_user = 'jupyter.experimenter@gmail.com'
        self.gmail_password = 'experimenter@1212'
        self.dest = 'eduferreiraj@gmail.com'


    def append(self, configuration):
        self.configurations.append(configuration)

    def run(self):
        print(self.configurations)
        for index, experimenter in enumerate(self.configurations):
            try:
                experimenter.start()
                if exists(experimenter.result_file):
                    continue

                rec = experimenter.recommender()
                logger.info('initialize recommendation model {}'.format(experimenter.exp_name))
                rec.initialize()

                batch_training, batch_test = experimenter.get_batch_data()

                evaluator = Evaluator(rec)
                logging.info('batch pre-training before streaming input')
                evaluator.fit(
                    batch_training,
                    batch_test,
                    max_n_epoch=20
                )

                logging.info('incrementally predict, evaluate and update the recommender')
                logging.info("Abrindo arquivo {} ...".format(experimenter.result_file))
                with open(experimenter.result_file, 'w+') as f:
                    logging.info("Começando a gerar resultados ...")
                    for instance in evaluator.evaluate(experimenter.get_prequential_data()):
                        f.write(str(instance))
                experimenter.finish()
                logging.info("Arquivo {} completo.".format(experimenter.result_file))
                subject = "[REPORT] {}".format(experimenter.exp_name.replace("_", " "))
                report = experimenter.get_report()
                report += "\n> Overview [{}/{}]:".format(index + 1, len(self.configurations))
                for c in self.configurations:
                    report += c.get_status()
            except Exception as e:
                subject = '[EXCEPT] {}'.format(e)
                body = 'Ooops, something happened in the experimentation.\n\nException:{}\n\n{}'.format(traceback.format_exc(), datetime.datetime.now())
                self.send_email(subject, body)
           # else:
           #     self.send_email(subject, report)

    def send_email(self, subject, body):
        sent_from = self.gmail_user
        sent_to = self.dest
        message = "From: {}\nTo: {}\nMIME-Version: 1.0\nSubject: {}\n{}".format(sent_from, sent_to, subject, body)
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(self.gmail_user, self.gmail_password)
        server.sendmail(sent_from, sent_to, message)
        server.close()
        logger.info('Email sent!')



if __name__ == "__main__":
    # Absolute
    BASE_PATH = "/home/eduardo/recsys/"

    # Relatives
    LOG_PATH = 'flurs/log/'


    RECALL_AT = 10

    logger = logging.getLogger("experimenter.jupyter")
    logging.basicConfig(format='%(name)-12s: %(levelname)-8s %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info('running %s' % ' '.join(sys.argv))

    data_paths = [
        "datasets/Protocol/ksampling/ciaodvd-gte.csv",
        "datasets/Protocol/ksampling/eachmovie-gte.csv",
        "datasets/Protocol/ksampling/ml-1m-gte.csv",
        "datasets/Protocol/ksampling/ml-100k-gte.csv",
        "datasets/Protocol/ksampling/movietweetings-gte.csv",
        "datasets/Protocol/ksampling/music-playlist.csv"
    ]
    t0 = 1
    tn = .001

    cooling_functions = [
        lambda i, N: 1 - i*(t0 - tn)/N,
        lambda i, N: (t0 - tn) / (1 + np.exp(.3 * (i - N / 2))) + tn,
        lambda i, N: t0*(tn/t0)**(i/N),
        lambda i, N: i ** (np.log(tn) / np.log(N)),
        lambda i, N: t0 * np.exp(-(1 / N) * np.log(t0 / tn) * i),
        lambda i, N: t0 * np.exp(-(1 / N ** 2) * np.log(t0 / tn) * i ** 2)
    ]

    k_values = list(range(2, 15))

    # for path in data_paths:
    #     generate_k_samples(path, k_values)

    k_samples_path = []

    for path in data_paths:
        for k_value in k_values:
            k_samples_path.append("{}-K{}.csv".format(path.split(".")[0], k_value))


    experimenter = Experimenter()
    learn_rate=.02
    ks = [40, 60]
    deltas = [2, 4, 8, 10]
    Ns = [10, 25, 75, 150]
    for k in ks:
        for scheme_number, cooling in enumerate(cooling_functions):
            for N in Ns:
                for delta in deltas:
                    experimenter.append(Configuration(BASE_PATH + path, MFRecommender, k=k, meta=CoolingLearning(N, delta, cooling), seed=s, exp_name="MFRecommender_Cooling_Scheme-{}_N{}_D{}".format(scheme_number + 1, N, delta)))

    experimenter.run()
