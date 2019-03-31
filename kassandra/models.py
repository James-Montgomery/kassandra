import scipy.stats as st
import numpy as np
#np.random.seed(592)

import tensorflow as tf
import progressbar
#tf.set_random_seed(2)

from keras.regularizers import l2

from . import build_utils

# https://github.com/keras-team/keras/issues/9412
# https://github.com/yaringal/DropoutUncertaintyExps/blob/master/net/net.py

# ---------------------------------------------------------------------------------------------------------------------
# Base Class
# ---------------------------------------------------------------------------------------------------------------------

class NeuralNetwork(object):
    """Neural Network Base Class"""

    def __init__(self, num_features=1, num_outputs=1, layers=2, units=50, activation=tf.nn.relu, dropout_rate=0.5, optimizer=None):

        # inputs and outputs
        self.num_features = num_features
        self.num_outputs = num_outputs

        # build specifications
        self.layers = layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.units = units

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

    def train(self, x_, y_, epochs=100, show_progress=True, show_loss=False):

        # training data
        feed_dict = {self.tensors["x"] : x_.reshape(-1, self.num_features),
                     self.tensors["y"] : y_.reshape(-1, self.num_outputs)}

        # set up progress bar
        if show_progress:
            pbar = progressbar.ProgressBar()
            n_iter = pbar(range(epochs))
        else:
            n_iter = range(epochs)

        # start training
        for i in n_iter:
            l, _ = self.session.run([self.tensors["loss"], self.step_op], feed_dict)

            if show_loss and i % 1000 ==0:
                print("loss {}".format(l))

# ---------------------------------------------------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------------------------------------------------

class MLP(NeuralNetwork):
    """Multi-Layer Perceptron"""

    def __init__(self, *args, **kwargs):
        self.name = "MLP"
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

    def __init__(self, lengthscale=None, *args, **kwargs):
        self.name = "BNDropout"

        if lengthscale is not None:
            N = 100
            reg = lengthscale**2 * (1 - self.dropout_rate) / (2. * N * tau)
            self.reg = l2(reg)
        else:
            self.reg = None

        NeuralNetwork.__init__(self, *args, **kwargs)

    def _build(self):
        return build_utils.build_bn_dropout(self)

    def predict(self, x_, n_estimates=10):
        feed_dict = {self.tensors["x"] : x_[:, np.newaxis]}

        samples = np.asarray([
            self.session.run([self.tensors["mu"], self.tensors["sigma"]], feed_dict)
            for i in range(n_estimates)
        ]).squeeze()

        mu, sigma = samples[:,0,:].mean(axis=0), samples[:,1,:].mean(axis=0)

        return np.asarray([mu, sigma]).squeeze()

    def sample_posterior(self, x_):
        feed_dict = {self.tensors["x"] : x_[:, np.newaxis]}
        posterior_sample = self.session.run([self.tensors["mu"], self.tensors["sigma"]], feed_dict)
        return np.asarray(posterior_sample).squeeze()

    def sample_posterior_predictive(self, x_):
        posterior_sample = self.sample_posterior(x_)
        mean, cov = posterior_sample[0,:], np.eye(posterior_sample.shape[1]) * posterior_sample[1,:]
        return st.multivariate_normal(mean=mean, cov=cov).rvs(1)

# ---------------------------------------------------------------------------------------------------------------------
# BNVI
# ---------------------------------------------------------------------------------------------------------------------
