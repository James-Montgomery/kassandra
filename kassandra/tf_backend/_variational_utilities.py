import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions

import logging as logging

tf.get_logger().setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################################################################

class NegativeLogLikelihood(tf.keras.losses.Loss):

    # default reduction is now SUM instead of auto
    def __init__(self, reduction=tf.keras.losses.Reduction.SUM,
                 name='vi_corrected_negative_log_likelihood'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y, p_y):
        return -p_y.log_prob(y)

################################################################################

class Dropout(tf.keras.layers.Dropout):
    """Applies Dropout to the input.
    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.
    # Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.
        noise_shape: 1D integer tensor representing the shape of the
            binary dropout mask that will be multiplied with the input.
            For instance, if your inputs have shape
            `(batch_size, timesteps, features)` and
            you want the dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.
    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
           http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
    """
    def __init__(self, rate, training=None, noise_shape=None, seed=None, **kwargs):
        super(Dropout, self).__init__(rate, noise_shape=None, seed=None,**kwargs)
        self.training = training


    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed)
            if not training:
                return K.in_train_phase(dropped_inputs, inputs, training=self.training)
            return K.in_train_phase(dropped_inputs, inputs, training=training)
        return inputs

################################################################################
# Independant Gaussians

def gaussian_variational_distribution(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
      tfp.layers.VariableLayer(2*n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t[..., :n],
                     scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
          reinterpreted_batch_ndims=1)),
    ])

def fixed_gaussian_prior_distribution_wrapper(mu=0.0, std=1.0):
    def fixed_gaussian_prior_distribution(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
          tfp.layers.VariableLayer(n, dtype=dtype),
          tfp.layers.DistributionLambda(lambda t: tfd.Independent(
              # use t*0.0 syntax to avoid errors when kl_use_exact=True
              tfd.Normal(loc=t*0.0+mu, scale=std),
              reinterpreted_batch_ndims=1)),
        ])
    return fixed_gaussian_prior_distribution

def mean_trainable_gaussian_prior_distribution_wrapper(std=1.0):
    def mean_trainable_gaussian_prior_distribution(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
          tfp.layers.VariableLayer(n, dtype=dtype),
          tfp.layers.DistributionLambda(lambda t: tfd.Independent(
              tfd.Normal(loc=t, scale=std),
              reinterpreted_batch_ndims=1)),
        ])
    return mean_trainable_gaussian_prior_distribution

# def std_trainable_gaussian_prior_distribution(kernel_size, bias_size=0, dtype=None):
#     n = kernel_size + bias_size
#     c = np.log(np.expm1(1.))
#     return tf.keras.Sequential([
#       tfp.layers.VariableLayer(n, dtype=dtype),
#       tfp.layers.DistributionLambda(lambda t: tfd.Independent(
#           tfd.Normal(loc=0.0,
#                      scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
#           reinterpreted_batch_ndims=1)),
#     ])

# def trainable_gaussian_prior_distribution(kernel_size, bias_size=0, dtype=None):
#     n = kernel_size + bias_size
#     c = np.log(np.expm1(1.))
#     return tf.keras.Sequential([
#       tfp.layers.VariableLayer(2*n, dtype=dtype),
#       tfp.layers.DistributionLambda(lambda t: tfd.Independent(
#           tfd.Normal(loc=t[..., :n],
#                      scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
#           reinterpreted_batch_ndims=1)),
#     ])

################################################################################
