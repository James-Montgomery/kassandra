from kassandra import utils

import scipy.stats as st
import numpy as np
import progressbar
import logging

import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None # remove stupid tf contrib warning

# ---------------------------------------------------------------------------------------------------------------------
# Base Class
# ---------------------------------------------------------------------------------------------------------------------

supported_activation_functions = {
    "relu": tf.nn.relu,
    "tanh": tf.nn.tanh
}


class NeuralNetwork(object):
    """Neural Network Base Class"""

    def __init__(self, num_features=1, num_outputs=1, layers=2, units=50, activation="relu", optimizer=None):

        # TODO: Add underscore to hidden variables

        # inputs and outputs
        self._num_features = num_features
        self._num_outputs = num_outputs

        # build specifications
        self._layers = layers
        self._units = units
        self._reg = None

        try:
            self._activation = supported_activation_functions[activation]
        except Exception as e:
            logging.error(e)
            raise Exception("{} is not a supported activation function.".format(activation))

        # set up the graph
        tf.reset_default_graph()
        self.tensors = self._build()
        # self.prior_tensors = self._build()
        self.session = tf.Session()

        # training specifications
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self._step_op = optimizer.minimize(self.tensors["loss"])

        # start session
        # TODO: Separate session for prior and regular tensors
        self.session.run(tf.global_variables_initializer())

        self._fitted = False

    def fit(self, x_train, y_train, epochs=100, show_progress=True, show_loss=False, shuffle=False, minibatches=1):

        N = x_train.shape[0]

        if shuffle:
            x_train, y_train = utils.shuffle(x_train, y_train, N)

        # set up progress bar
        if show_progress:
            pbar = progressbar.ProgressBar()
            n_iter = pbar(range(epochs))
        else:
            n_iter = range(epochs)

        batch_size = int(N / minibatches)
        assert batch_size > 0, "Specified too many mini-batches for number of train samples."

        # start training
        with utils.Timer():
            for i in n_iter:

                batch_idx = np.random.choice(N, batch_size, replace=False)

                feed_dict = {self.tensors["x"] : x_train[batch_idx].reshape(-1, self._num_features),
                             self.tensors["y"] : y_train[batch_idx].reshape(-1, self._num_outputs)}

                l, _ = self.session.run([self.tensors["loss"], self._step_op], feed_dict)

                if show_loss and i % 1000 ==0:
                    print("loss {}".format(l))

        self._fitted = True

    ##
    # Get Prior / Posterior / Posterior Predictive
    ##

    def get_prior(self, x_test, n_estimates=10, return_cov=False):
        feed_dict = {self.tensors["x"]: x_test[:, np.newaxis]}

        samples = np.asarray([
            self.session.run(self.prior_tensors["mu"], feed_dict)
            for _ in range(n_estimates)
        ]).squeeze()

        mean = samples.mean(axis=0)
        var = samples.std(axis=0)**2

        if not return_cov:
            return mean, var

        cov = np.cov(samples[:, 0, :].T)
        return mean, cov

    def get_prior_predictive(self, x_test, n_estimates=10, return_cov=False):
        feed_dict = {self.prior_tensors["x"]: x_test[:, np.newaxis]}

        samples = np.asarray([
            self.session.run([self.prior_tensors["mu"], self.prior_tensors["sigma"]], feed_dict)
            for i in range(n_estimates)
        ]).squeeze()

        mean = samples[:, 0, :].mean(axis=0)
        var = samples[:, 0, :].std(axis=0)**2
        var_l = samples[:, 1, :].mean(axis=0)**2

        if not return_cov:
            return mean, var + var_l

        cov = np.cov(samples[:, 0, :].T)
        identity = np.zeros(cov.shape)
        idx = np.diag_indices(identity.shape[0])
        identity[idx] = var_l
        cov += identity

        return mean, cov

    def get_posterior(self, x_test, n_estimates=10, return_cov=False):

        assert n_estimates >= 1, "Invalid argument for n_estimators. Must be integer >= 1"

        feed_dict = {self.tensors["x"]: x_test[:, np.newaxis]}

        samples = np.asarray([
            self.session.run(self.tensors["mu"], feed_dict)
            for _ in range(n_estimates)
        ]).squeeze()

        mean = samples.mean(axis=0)
        var = samples.std(axis=0)**2

        if not return_cov:
            return mean, var

        cov = np.cov(samples[:, 0, :].T)
        return mean, cov

    def get_posterior_predictive(self, x_test, n_estimates=10, return_cov=False):

        assert n_estimates >= 1, "Invalid argument for n_estimators. Must be integer >= 1"

        feed_dict = {self.tensors["x"]: x_test[:, np.newaxis]}

        samples = np.asarray([
            self.session.run([self.tensors["mu"], self.tensors["sigma"]], feed_dict)
            for _ in range(n_estimates)
        ]).squeeze()

        mean = samples[:, 0, :].mean(axis=0)
        var = samples[:, 0, :].std(axis=0)**2
        var_l = samples[:, 1, :].mean(axis=0)**2

        if not return_cov:
            return mean, var + var_l

        cov = np.cov(samples[:, 0, :].T)
        identity = np.zeros(cov.shape)
        idx = np.diag_indices(identity.shape[0])
        identity[idx] = var_l
        cov += identity

        return mean, cov

    ##
    # Sample Prior / Posterior / Posterior Predictive
    ##

    def sample_prior(self, x_test):
        feed_dict = {self.prior_tensors["x"]: x_test[:, np.newaxis]}
        posterior_sample = self.session.run(self.prior_tensors["mu"], feed_dict)
        return np.asarray(posterior_sample)

    def sample_prior_predictive(self, x_test):
        feed_dict = {self.prior_tensors["x"]: x_test[:, np.newaxis]}
        posterior_sample = np.asarray(self.session.run([self.prior_tensors["mu"], self.prior_tensors["sigma"]], feed_dict))
        mean = posterior_sample[0, :, 0]
        cov = np.eye(posterior_sample.shape[1]) * posterior_sample[1, :, 0]
        return st.multivariate_normal(mean=mean, cov=cov).rvs(1)

    def sample_posterior(self, x_test):
        feed_dict = {self.tensors["x"]: x_test[:, np.newaxis]}
        posterior_sample = self.session.run(self.tensors["mu"], feed_dict)
        return np.asarray(posterior_sample)

    def sample_posterior_predictive(self, x_test):
        feed_dict = {self.tensors["x"]: x_test[:, np.newaxis]}
        posterior_sample = np.asarray(self.session.run([self.tensors["mu"], self.tensors["sigma"]], feed_dict))
        mean = posterior_sample[0, :, 0]
        cov = np.eye(posterior_sample.shape[1]) * posterior_sample[1, :, 0]
        return st.multivariate_normal(mean=mean, cov=cov).rvs(1)

    ##
    # Get Log Probabilities
    ##

    def get_prior_logpdf(self, x_test):
        pass

    def get_posterior_logpdf(self, x_test):
        pass

# ---------------------------------------------------------------------------------------------------------------------
# Regression Class
# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------
# Homoscedastic Class
# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------
# Heteroscedastic Class
# ---------------------------------------------------------------------------------------------------------------------
