from kassandra import build_utils
from kassandra.base_model import NeuralNetwork

import numpy as np
import logging

import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None # remove stupid tf contrib warning
tf.logging.set_verbosity(tf.logging.ERROR)

# ---------------------------------------------------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------------------------------------------------


class MLP(NeuralNetwork):
    """Multi-Layer Perceptron"""

    def __init__(self, dropout_rate=0.5, *args, **kwargs):
        self.name = "MLP"
        self._dropout_rate = dropout_rate
        NeuralNetwork.__init__(self, *args, **kwargs)

    def _build(self):
        return build_utils.build_mlp(self)

    def predict(self, x_test):
        if not self._fitted:
            logging.warning("Making a prediction on an unfitted model.")

        feed_dict = {self.tensors["x"] : x_test[:, np.newaxis]}
        return np.asarray(self.session.run(self.tensors["mu"], feed_dict) ).squeeze()

    def get_posterior(self):
        raise Exception("Not Supported for Non-Bayesian Neural Network")

    def get_posterior_predictive(self):
        raise Exception("Not Supported for Non-Bayesian Neural Network")

    def sample_posterior(self):
        raise Exception("Not Supported for Non-Bayesian Neural Network")

    def sample_posterior_predictive(self):
        raise Exception("Not Supported for Non-Bayesian Neural Network")

# ---------------------------------------------------------------------------------------------------------------------
# BNDropout
# ---------------------------------------------------------------------------------------------------------------------


class BNDropout(NeuralNetwork):
    """Bayesian Neural Network using MC Dropout"""

    def __init__(self, dropout_rate=0.5, num_train_samples=100, lengthscale=None, *args, **kwargs):
        self.name = "BNDropout"
        self._dropout_rate = dropout_rate
        self._num_train_samples = num_train_samples
        self._lengthscale = lengthscale

        NeuralNetwork.__init__(self, *args, **kwargs)

    def _build(self):
        return build_utils.build_bn_dropout(self)

# ---------------------------------------------------------------------------------------------------------------------
# BNVI
# ---------------------------------------------------------------------------------------------------------------------


class BNVI(NeuralNetwork):
    """Bayesian Neural Network using Variational Inference"""

    def __init__(self, num_train_samples=100, p_mean=0.0, p_std=1.0, q_mean=0.0, q_std=1.0, *args, **kwargs):
        self.name = "BNVI"
        self._num_train_samples = num_train_samples

        # prior parameters
        self.p_mean = p_mean
        self.p_std = p_std

        # variational parameters
        self.q_mean = q_mean
        self.q_std = q_std

        NeuralNetwork.__init__(self, *args, **kwargs)

    def _build(self):
        return build_utils.build_bn_vi(self)
