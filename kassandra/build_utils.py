import numpy as np
#np.random.seed(592)

import tensorflow as tf
#tf.set_random_seed(2)

# ---------------------------------------------------------------------------------------------------------------------
# Base Function
# ---------------------------------------------------------------------------------------------------------------------

def build(self, bayes):

    x = tf.placeholder(shape=(None, self.num_features), dtype=tf.float32, name="x")
    y = tf.placeholder(shape=(None, self.num_outputs), dtype=tf.float32, name="y")

    d = x
    for layer in range(self.layers):
        layer +=1

        l = tf.layers.dense(inputs=d,
                            units=self.units,
                            activation=self.activation,
                            kernel_regularizer=self.reg,
                            name="layer_{}".format(layer),
                            reuse=tf.AUTO_REUSE)

        d = tf.layers.dropout(inputs=l,
                              rate=self.dropout_rate,
                              noise_shape = (1, self.units),
                              training=bayes,
                              name="dropout_{}".format(layer))

        return x, y, d

# ---------------------------------------------------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------------------------------------------------

def build_mlp(self):

    x, y, d = build(self, False)

    mu = tf.layers.dense(inputs=d,
                            units=self.num_outputs,
                            activation=None,
                            name="mu",
                            reuse=tf.AUTO_REUSE)

    loss = tf.losses.mean_squared_error(y, mu)
    if self.reg is not None:
        loss += tf.losses.get_regularization_losses()

    return {
        "x" : x,
        "y" : y,
        "mu" : mu,
        "loss" : loss
    }

# ---------------------------------------------------------------------------------------------------------------------
# BNDropout
# ---------------------------------------------------------------------------------------------------------------------

def build_bn_dropout(self):

    if self.lengthscale is not None:
        tau = 1.0
        N = self.train_samples
        reg = self.lengthscale**2 * (1 - self.dropout_rate) / (2. * N * tau)
        self.reg = tf.contrib.layers.l2_regularizer(reg)

    x, y, d = build(self, True)

    mu = tf.layers.dense(inputs=d,
                            units=self.num_outputs,
                            activation=None,
                            kernel_regularizer=self.reg,
                            name="mu",
                            reuse=tf.AUTO_REUSE)

    sigma = tf.layers.dense(inputs=d,
                            units=self.num_outputs,
                            activation=tf.nn.softplus,
                            kernel_regularizer=self.reg,
                            name="sigma",
                            reuse=tf.AUTO_REUSE)

    likelihood = tf.distributions.Normal(loc=mu, scale=sigma)

    loss = tf.reduce_mean(-likelihood.log_prob(y))
    if self.reg is not None:
        loss += tf.losses.get_regularization_losses()

    return {
        "x" : x,
        "y" : y,
        "mu" : mu,
        "sigma" : sigma,
        "loss" : loss
    }
