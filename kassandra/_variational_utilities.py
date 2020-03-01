import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import logging as logging

tf.get_logger().setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################################################################

class VICorrectedNegativeLogLikelihood(tf.keras.losses.Loss):

    # default reduction is now SUM instead of auto
    def __init__(self, reduction=tf.keras.losses.Reduction.SUM,
                 name='corrected_negative_log_likelihood'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y, p_y):
        return -p_y.log_prob(y)

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
