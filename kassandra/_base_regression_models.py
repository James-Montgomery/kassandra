import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

import tensorflow_probability as tfp
tfd = tfp.distributions

import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class AleatoricLoss(object):

    def __init__(self):
        pass

    def __call__(self, mean, sigma, y_true):
        likelihood_distribution = tfd.Normal(loc=mean, scale=sigma)
        return tf.reduce_mean(-likelihood_distribution.log_prob(y_true)) # negative log likelihood


def check_aleatoric_uncertainty(aleatoric_uncertainty, init_homoscedastic_noise):

    valid_aleatoric_uncertainties = [None, "homoscedastic", "heteroscedastic"]

    if aleatoric_uncertainty not in valid_aleatoric_uncertainties:
        raise Exception("Invalid argument for aleatoric_uncertainty. "
                        "Please try: {}".format(valid_aleatoric_uncertainties))

    if aleatoric_uncertainty == "homoscedastic" and init_homoscedastic_noise is None:
        init_homoscedastic_noise = 10.0
        logger.warning("init_homoscedastic_noise not specified. Setting to {}".format(init_homoscedastic_noise))

    return aleatoric_uncertainty, init_homoscedastic_noise

# ---------------------------------------------------------------------------------------------------------------------
# Dropout Model Class
# ---------------------------------------------------------------------------------------------------------------------


class _DropoutModel(tf.keras.Model):

    def __init__(self, layers=0, units=1, activation='relu', dropout_rate=0,
                 mc_dropout=False, aleatoric_uncertainty=None,
                 init_kernel_magnitude=0.5, init_bias_magnitude=10.0, init_homoscedastic_noise=None):

        super(_DropoutModel, self).__init__()

        self.aleatoric_uncertainty, init_homoscedastic_noise = check_aleatoric_uncertainty(aleatoric_uncertainty,
                                                                                           init_homoscedastic_noise)

        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0,
                                                                stddev=init_kernel_magnitude)

        bias_initializer = tf.keras.initializers.RandomNormal(mean=0.0,
                                                              stddev=init_bias_magnitude)

        self.neural_net = []

        for _ in range(layers):
            dense_layer = Dense(units=units,
                                activation=activation,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer)
            self.neural_net.append(dense_layer)

            if dropout_rate > 0:
                dropout_layer = lambda x: Dropout(rate=dropout_rate,
                                                  noise_shape=(1, units))(x, training=mc_dropout)
                self.neural_net.append(dropout_layer)

        if self.aleatoric_uncertainty is None:
            mean = Dense(units=1,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer)

            self.neural_net.append(mean)

        elif self.aleatoric_uncertainty == "heteroscedastic":
            mean = Dense(units=1,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer)

            sigma = Dense(units=1,
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          activation="softplus")

            self.neural_net.append(mean)
            self.neural_net.append(sigma)

        elif self.aleatoric_uncertainty == "homoscedastic":
            mean = Dense(units=1,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer)

            sigma = tf.keras.activations.softplus( tf.Variable(init_homoscedastic_noise*tf.ones(shape=(1), dtype=tf.float64), name="simga") )

            self.neural_net.append(mean)
            self.neural_net.append(sigma)

        else:
            raise Exception("Invalid value for aleatoric_uncertainty: {}".format(self.aleatoric_uncertainty))

    def get_loss_object(self):

        if self.aleatoric_uncertainty is None:
            return tf.keras.losses.MeanSquaredError()
        else:
            return AleatoricLoss()

    def __call__(self, x):

        if self.aleatoric_uncertainty is None:

            for layer in self.neural_net:
                x = layer(x)

            return x

        elif self.aleatoric_uncertainty is "heteroscedastic":

            for layer in self.neural_net[:-2]:
                x = layer(x)

            mean = self.neural_net[-2](x)

            sigma = self.neural_net[-1](x)

            return mean, sigma

        elif self.aleatoric_uncertainty == "homoscedastic":

            for layer in self.neural_net[:-2]:
                x = layer(x)

            mean = self.neural_net[-2](x)

            sigma = self.neural_net[-1]

            return mean, tf.tile(sigma[:, tf.newaxis], [mean.shape[0], 1])

        raise Exception("Invalid value for aleatoric_uncertainty: {}".format(self.aleatoric_uncertainty))

# ---------------------------------------------------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------------------------------------------------

