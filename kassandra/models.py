import scipy.stats as st
import numpy as np
#np.random.seed(592)

import tensorflow as tf
import progressbar
#tf.set_random_seed(2)

from . import build_utils
from . import utils

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------------------------------------------------
# Base Class
# ---------------------------------------------------------------------------------------------------------------------

class NeuralNetwork(object):
    """Neural Network Base Class"""

    def __init__(self, num_features=1, num_outputs=1, layers=2, units=50, activation=tf.nn.relu, optimizer=None):

        # inputs and outputs
        self.num_features = num_features
        self.num_outputs = num_outputs

        # build specifications
        self.layers = layers
        self.activation = activation
        self.units = units
        self.reg = None

        # set up the graph
        tf.reset_default_graph()
        self.tensors = self._build()
        self.session = tf.Session()

        # training specifications
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.step_op = optimizer.minimize(self.tensors["loss"])

        # start session
        self.session.run(tf.global_variables_initializer())

    def train(self, x_train, y_train, epochs=100, show_progress=True, show_loss=False, shuffle=False, minibatches=1):

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
        assert batch_size > 0, "Specified too many minibatches for number of train samples."

        # start training
        with utils.Timer():
            for i in n_iter:

                batch_idx = np.random.choice(N, batch_size, replace=False)

                feed_dict = {self.tensors["x"] : x_train[batch_idx].reshape(-1, self.num_features),
                            self.tensors["y"] : y_train[batch_idx].reshape(-1, self.num_outputs)}

                l, _ = self.session.run([self.tensors["loss"], self.step_op], feed_dict)


                if show_loss and i % 1000 ==0:
                    print("loss {}".format(l))

# ---------------------------------------------------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------------------------------------------------

class MLP(NeuralNetwork):
    """Multi-Layer Perceptron"""

    def __init__(self, dropout_rate=0.5, *args, **kwargs):
        self.name = "MLP"
        self.dropout_rate = dropout_rate
        NeuralNetwork.__init__(self, *args, **kwargs)

    def _build(self):
        return build_utils.build_mlp(self)

    def predict(self, x_, n_estimates=10):
        feed_dict = {self.tensors["x"] : x_[:, np.newaxis]}
        return np.asarray(self.session.run(self.tensors["mu"], feed_dict) ).squeeze()

# ---------------------------------------------------------------------------------------------------------------------
# BNDropout
# ---------------------------------------------------------------------------------------------------------------------

class BNDropout(NeuralNetwork):
    """Bayesian Neural Network using MC Dropout"""

    def __init__(self, dropout_rate=0.5, train_samples=100, lengthscale=None, *args, **kwargs):
        self.name = "BNDropout"
        self.dropout_rate = dropout_rate
        self.train_samples = train_samples
        self.lengthscale = lengthscale

        NeuralNetwork.__init__(self, *args, **kwargs)

    def _build(self):
        return build_utils.build_bn_dropout(self)

    def predict(self, x_test, n_estimates=10):
        feed_dict = {self.tensors["x"] : x_test[:, np.newaxis]}

        samples = np.asarray([
            self.session.run([self.tensors["mu"], self.tensors["sigma"]], feed_dict)
            for i in range(n_estimates)
        ]).squeeze()

        mu, sigma = samples[:,0,:].mean(axis=0), samples[:,1,:].mean(axis=0)

        return np.asarray([mu, sigma]).squeeze()

    def sample_posterior(self, x_test):
        feed_dict = {self.tensors["x"] : x_test[:, np.newaxis]}
        posterior_sample = self.session.run([self.tensors["mu"], self.tensors["sigma"]], feed_dict)
        return np.asarray(posterior_sample).squeeze()

    def sample_posterior_predictive(self, x_test):
        posterior_sample = self.sample_posterior(x_test)
        mean, cov = posterior_sample[0,:], np.eye(posterior_sample.shape[1]) * posterior_sample[1,:]
        return st.multivariate_normal(mean=mean, cov=cov).rvs(1)

# ---------------------------------------------------------------------------------------------------------------------
# BNVI
# ---------------------------------------------------------------------------------------------------------------------

class BNVI(NeuralNetwork):
    """Bayesian Neural Network using Variational Inference"""

    def __init__(self, train_samples=100, p_mean=0.0, p_std=1.0, q_mean=0.0, q_std=1.0, *args, **kwargs):
        self.name = "BNVI"
        self.train_samples = train_samples

        # prior parameters
        self.p_mean = p_mean
        self.p_std = p_std

        # variational parameters
        self.q_mean = q_mean
        self.q_std = q_std

        NeuralNetwork.__init__(self, *args, **kwargs)

    def _build(self):
        return build_utils.build_bn_vi(self)

    def predict(self, x_test, n_estimates=10):
        feed_dict = {self.tensors["x"] : x_test[:, np.newaxis]}

        samples = np.asarray([
            self.session.run([self.tensors["mu"], self.tensors["sigma"]], feed_dict)
            for i in range(n_estimates)
        ]).squeeze()

        mu, sigma = samples[:,0,:].mean(axis=0), samples[:,1,:].mean(axis=0)

        return np.asarray([mu, sigma]).squeeze()

    def sample_posterior(self, x_test):
        feed_dict = {self.tensors["x"] : x_test[:, np.newaxis]}
        posterior_sample = self.session.run([self.tensors["mu"], self.tensors["sigma"]], feed_dict)
        return np.asarray(posterior_sample).squeeze()

    def sample_posterior_predictive(self, x_test):
        posterior_sample = self.sample_posterior(x_test)
        mean, cov = posterior_sample[0,:], np.eye(posterior_sample.shape[1]) * posterior_sample[1,:]
        return st.multivariate_normal(mean=mean, cov=cov).rvs(1)
