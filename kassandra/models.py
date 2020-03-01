import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import logging as logging

tf.get_logger().setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # when installed
    from kassandra import _base_class
    from kassandra import _variational_utilities as _vu
except:
    # for local testing
    from . import _base_class
    from . import _variational_utilities as _vu

################################################################################

def set_tf_seed(seed=49):
    tf.random.set_seed(seed)

################################################################################

REGULARIZERS = {
    "l1": "l1",
    "l2": "l2",
}

WEIGHT_INITIALIZER = {
    "glorot_normal": "glorot_normal",
    "glorot_uniform": "glorot_uniform"
}

VARIATIONAL_DISTRIBUTIONS = {
    "independant_gaussians": "independant_gaussians"
}
PRIOR_DISTRIBUTIONS = VARIATIONAL_DISTRIBUTIONS

DISTRIBUTION_ATTRIBUTES = {
    "independant_gaussians": ["mu", "std"]
}

################################################################################

class MLEMLP(_base_class.NeuralNetwork):
    """
    Maximum Likelihood Estimate for Multi-Layer Perceptron
    """

    def __init__(self, regularization=None, weight_initializer="glorot_normal", *args, **kwargs):

        super(MLEMLP, self).__init__(*args, **kwargs)

        self._regularization = None if regularization is None \
            else REGULARIZERS[regularization]
        self._weight_initializer = WEIGHT_INITIALIZER[weight_initializer]

        self.model = self._build()
        self.model.compile(optimizer=self._optimizer, loss=self._loss)

    def _build(self):

        layers = []

        # TODO: when using l2 regularization, do we need to chenge to
        # SUM reduction for the loss functions?

        # build hidden layers
        for i in range(self._num_hidden_layers):
            layers.append(
                tf.keras.layers.Dense(self._num_hidden_units,
                              kernel_initializer=self._weight_initializer,
                              bias_initializer=self._weight_initializer,
                              activation=self._activation,
                              kernel_regularizer=self._regularization,
                              bias_regularizer=self._regularization)
            )

        # classiffiation
        if self._regression_flag is False:

            # TODO: do we regularize the last layer?

            # multiclass
            if self._output_dim > 2:
                self._loss = tf.keras.losses.CategoricalCrossentropy()
                layers.append(tf.keras.layers.Dense(self._output_dim,
                                activation="softmax",
                                kernel_regularizer=self._regularization,
                                bias_regularizer=self._regularization))
            # binary
            else:
                self._loss = tf.keras.losses.BinaryCrossentropy()
                layers.append(tf.keras.layers.Dense(1,
                                activation="sigmoid",
                                kernel_regularizer=self._regularization,
                                bias_regularizer=self._regularization))

        # regression
        else:

            if self._aleatoric_flag is False:
                self._loss = tf.keras.losses.MeanSquaredError()
                layers.append(tf.keras.layers.Dense(self._output_dim,
                                activation=None,
                                kernel_regularizer=self._regularization,
                                bias_regularizer=self._regularization))

            else:

                self._loss = lambda y, p_y: -p_y.log_prob(y)

                if self._heteroskedastic is True:
                    layers.append(tf.keras.layers.Dense(self._output_dim*2,
                                    activation=None,
                                    kernel_regularizer=self._regularization,
                                    bias_regularizer=self._regularization))
                    layers.append(tfp.layers.DistributionLambda(
                          lambda t: tfd.Normal(loc=t[..., :self._output_dim],
                                               scale=1e-3+tf.math.softplus(0.05*t[..., self._output_dim:]))))

                else:
                    if self._aleatoric_stddev is not None:
                        layers.append(tf.keras.layers.Dense(self._output_dim,
                                    activation=None,
                                    kernel_regularizer=self._regularization,
                                    bias_regularizer=self._regularization))
                        layers.append(tfp.layers.DistributionLambda(lambda t:
                           tfd.Normal(loc=t, scale=self._aleatoric_stddev)))

                    else:
                        raise Exception("Currently not supported.")

        return tf.keras.Sequential(layers)

    def predict(self, x_test):

        if self._fitted is False:
            logger.warning("Predict called on a model that \
                has not been fit to data.")

        return self.model(x_test)

################################################################################

class VIMLP(_base_class.NeuralNetwork):
    """
    Variational Inference for Multi-Layer Perceptron
    """

    def __init__(self, variational_distribution="independant_gaussians",
                 prior_distribution="independant_gaussians",
                 prior_parameters={"mu": 0.0, "std": 1.0},
                 kl_use_exact=True, fixed_batch_size=None, *args, **kwargs):

        super(VIMLP, self).__init__(*args, **kwargs)

        self._variational_distribution = VARIATIONAL_DISTRIBUTIONS[variational_distribution]
        self._prior_distribution = PRIOR_DISTRIBUTIONS[prior_distribution]
        self._prior_parameters = prior_parameters
        self._kl_use_exact = kl_use_exact
        self._kl_weight = None if fixed_batch_size is None else 1.0 / fixed_batch_size
        self._reduction = "sum" if self._kl_weight is None else "auto"

        self.model = self._build()
        self.model.compile(optimizer=self._optimizer, loss=self._loss)

    def _build_vi_params(self):

        if self._variational_distribution == "independant_gaussians":
            variational_distribution = _vu.gaussian_variational_distribution
        else:
            raise Exception("Unsupported variational distribution: {}".format(
                self._variational_distribution))

        if self._prior_distribution == "independant_gaussians":
            mu, std = self._prior_parameters["mu"], self._prior_parameters["std"]
            if mu is not None and std is not None:
                prior_distribution = _vu.fixed_gaussian_prior_distribution_wrapper(mu=mu, std=std)
            elif mu is None and std is not None:
                prior_distribution = _vu.mean_trainable_gaussian_prior_distribution_wrapper(std=std)
            else:
                raise Exception("Not implemented yet!")
        else:
            raise Exception("Unsupported prior distribution: {}".format(
                self._prior_distribution))

        return variational_distribution, prior_distribution

    def _build(self):

        layers = []

        # define prior and variational distributions
        variational_distribution, prior_distribution = self._build_vi_params()

        # build hidden layers
        for i in range(self._num_hidden_layers):
            layers.append(
                tfp.layers.DenseVariational(3,
                                            variational_distribution,
                                            prior_distribution,
                                            kl_use_exact=self._kl_use_exact,
                                            kl_weight=self._kl_weight,
                                            activation=tf.keras.activations.relu)
            )

        # classiffiation
        if self._regression_flag is False:

            # multiclass
            if self._output_dim > 2:
                # keras sums KL regularization but tf averages loss
                self._loss = tf.keras.losses.CategoricalCrossentropy(reduction=self._reduction)
                layers.append(tfp.layers.DenseVariational(self._output_dim,
                                    variational_distribution,
                                    prior_distribution,
                                    kl_use_exact=self._kl_use_exact,
                                    kl_weight=self._kl_weight,
                                    activation="softmax"))

            # binary
            else:
                # keras sums KL regularization but tf averages loss
                self._loss = tf.keras.losses.BinaryCrossentropy(reduction=self._reduction)
                layers.append(tfp.layers.DenseVariational(1,
                                    variational_distribution,
                                    prior_distribution,
                                    kl_use_exact=self._kl_use_exact,
                                    kl_weight=self._kl_weight,
                                    activation="sigmoid"))

        # regression
        else:

            if self._aleatoric_flag is False:
                # keras sums KL regularization but tf averages loss
                self._loss = tf.keras.losses.MeanSquaredError(reduction=self._reduction)
                layers.append(tfp.layers.DenseVariational(self._output_dim,
                                    variational_distribution,
                                    prior_distribution,
                                    kl_use_exact=self._kl_use_exact,
                                    kl_weight=self._kl_weight,
                                    activation=None))

            else:

                # keras sums KL regularization but tf averages loss
                self._loss = _vu.VICorrectedNegativeLogLikelihood(reduction=self._reduction)
                # TODO: replace loss lambdas with this class

                if self._heteroskedastic is True:
                    layers.append(tfp.layers.DenseVariational(self._output_dim*2,
                                        variational_distribution,
                                        prior_distribution,
                                        kl_use_exact=self._kl_use_exact,
                                        kl_weight=self._kl_weight,
                                        activation=None))
                    layers.append(tfp.layers.DistributionLambda(
                          lambda t: tfd.Normal(loc=t[..., :self._output_dim],
                                               scale=1e-3+tf.math.softplus(0.05*t[..., self._output_dim:]))))

                else:
                    if self._aleatoric_stddev is not None:
                        layers.append(tfp.layers.DenseVariational(self._output_dim,
                                            variational_distribution,
                                            prior_distribution,
                                            kl_use_exact=self._kl_use_exact,
                                            kl_weight=self._kl_weight,
                                            activation=None))
                        layers.append(tfp.layers.DistributionLambda(lambda t:
                           tfd.Normal(loc=t, scale=self._aleatoric_stddev)))

                    else:
                        raise Exception("Currently not supported.")

        return tf.keras.Sequential(layers)

    def sample_posterior(self, x_test):

        if self._fitted is False:
            logger.warning("Sample posterior called on a model that \
                has not been fit to data. Not equivalent to sampling from \
                the true prior.")

        return self.model(x_test)

################################################################################

class DVIFMLP(_base_class.NeuralNetwork):
    """
    Dropout Variational Inference using Flipout for Multi-Layer Perceptron
    """

    def __init__(self, trainable_noise=True, *args, **kwargs):

        super(DVIFMLP, self).__init__(*args, **kwargs)

        self._trainable_noise = trainable_noise

        self.model = self._build()
        self.model.compile(optimizer=self._optimizer, loss=self._loss)

    def _build(self):

        layers = []

        # build hidden layers
        for i in range(self._num_hidden_layers):
            layers.append(
                tfp.layers.DenseFlipout(self._num_hidden_units,
                              trainable=self._trainable_noise,
                              activation=self._activation)
            )

        # classiffiation
        if self._regression_flag is False:

            # multiclass
            if self._output_dim > 2:
                self._loss = tf.keras.losses.CategoricalCrossentropy()
                layers.append(tf.keras.layers.Dense(self._output_dim,
                                activation="softmax"))
            # binary
            else:
                self._loss = tf.keras.losses.BinaryCrossentropy()
                layers.append(tf.keras.layers.Dense(1,
                                activation="sigmoid"))

        # regression
        else:

            if self._aleatoric_flag is False:
                self._loss = tf.keras.losses.MeanSquaredError()
                layers.append(tf.keras.layers.Dense(self._output_dim,
                                activation=None))

            else:

                self._loss = lambda y, p_y: -p_y.log_prob(y)

                if self._heteroskedastic is True:
                    layers.append(tf.keras.layers.Dense(self._output_dim*2,
                                    activation=None))
                    layers.append(tfp.layers.DistributionLambda(
                          lambda t: tfd.Normal(loc=t[..., :self._output_dim],
                                               scale=1e-3+tf.math.softplus(0.05*t[..., self._output_dim:]))))

                else:
                    if self._aleatoric_stddev is not None:
                        layers.append(tf.keras.layers.Dense(self._output_dim,
                                    activation=None))
                        layers.append(tfp.layers.DistributionLambda(lambda t:
                           tfd.Normal(loc=t, scale=self._aleatoric_stddev)))

                    else:
                        raise Exception("Currently not supported.")

        return tf.keras.Sequential(layers)

    def sample_posterior(self, x_test):

        if self._fitted is False:
            logger.warning("Sample posterior called on a model that \
                has not been fit to data. Not equivalent to sampling from \
                the true prior.")

        return self.model(x_test)

################################################################################

class MCMCMLP(_base_class.NeuralNetwork):
    """
    Markov Chain Monte Carlo for Multi-Layer Perceptron
    """

    def __init__(self):

        super(MCMCMLP, self).__init__(*args, **kwargs)

################################################################################
