"""
"""

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import logging as logging

tf.get_logger().setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from kassandra import _base_class
from kassandra.tf_backend import _tf_base_class
from kassandra.tf_backend import _variational_utilities as _vu
from kassandra.utilities import *

################################################################################

VARIATIONAL_DISTRIBUTIONS = {
    "independant gaussians": "independant gaussians",
    "mean field approximation": "independant gaussians",
}
PRIOR_DISTRIBUTIONS = VARIATIONAL_DISTRIBUTIONS

DISTRIBUTION_ATTRIBUTES = {
    "independant gaussians": ["mu", "std"],
    "mean field approximation": ["mu", "std"]
}

################################################################################

class VIMLP(_tf_base_class.TensorflowModel, _base_class.BayesianModel):
    """
    Variational Inference for Multi-Layer Perceptron
    """

    def __init__(self, variational_params=None,
                 kl_use_exact=True, fixed_batch_size=None,
                 *args, **kwargs):
        """
        """

        super(VIMLP, self).__init__(*args, **kwargs)

        if variational_params is None:
            variational_params = {
                "variational distribution": "independant gaussians",
                "prior distribtution": "independant gaussians",
                "prior parameters": {
                    "mu": 0.0,
                    "std": 1.0
                }
            }

        if variational_params["variational distribution"] not in VARIATIONAL_DISTRIBUTIONS.keys():
            raise ValueError("Invalid argument for variational distribution.")
        if variational_params["prior distribution"] not in PRIOR_DISTRIBUTIONS.keys():
            raise ValueError("Invalid argument for prior distribution.")

        self._variational_distribution = variational_params["variational distribution"]
        self._prior_distribution = variational_params["prior distribution"]
        self._prior_parameters = variational_params["prior parameters"]

        # make sure the kl divergence penalty is properly weighted
        self._kl_use_exact = kl_use_exact
        self._kl_weight = None if fixed_batch_size is None else \
            1.0 / fixed_batch_size
        self._reduction = "sum" if self._kl_weight is None else "auto"

        self.model = self._build()
        self.model.compile(optimizer=self._optimizer, loss=self._loss)

    def get_prior(self, x_test):
        """
        """
        check_array_input(x_test, "x_test")
        super().get_prior()

    def sample_prior(self, x_test):
        """
        """
        check_array_input(x_test, "x_test")
        super().sample_prior()

    def get_prior_predictive(self, x_test):
        """
        """
        check_array_input(x_test, "x_test")
        super().get_prior_predictive()

    def sample_prior_predictive(self, x_test):
        """
        """
        check_array_input(x_test, "x_test")
        super().sample_prior_predictive()

    def _build_vi_params(self):
        """
        """

        # build variational distribution
        if self._variational_distribution == "independant gaussians":
            variational_distribution = _vu.gaussian_variational_distribution
        else:
            raise Exception("Unsupported variational distribution: {}".format(
                self._variational_distribution))

        # build prior distribution
        if self._prior_distribution == "independant gaussians":
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

    def _get_output_params(self):

        if self._likelihood == "bernoulli": # binary classification
            output_activation = "sigmoid"
            self._loss = tf.keras.losses.BinaryCrossentropy(reduction=self._reduction)
        elif self._likelihood == "categorical": # multiclass classification
            output_activation = "softmax"
            loss = tf.keras.losses.CategoricalCrossentropy(reduction=self._reduction)
        else: # regression
            output_activation = None
            loss = _vu.NegativeLogLikelihood(reduction=self._reduction)

        self._loss = loss
        self._output_activation = output_activation

    def _get_output_dimensions(self):

        if self._likelihood == "bernoulli": # binary classification
            return self._output_dim
        elif self._likelihood == "categorical": # multiclass classification
            return self._output_dim
        elif "gaussian":
            mu = self._likelihood_parameters["mu"]
            std = self._likelihood_parameters["std"]
        elif "student-t":
            mu = self._likelihood_parameters["mu"]
            std = self._likelihood_parameters["std"]
            nu = self._likelihood_parameters["nu"]
        else:
            raise ValueError("Unsupported Likelihood distribution.")

    def _build(self):
        """
        """

        layers = []

        # define prior and variational distributions
        variational_distribution, prior_distribution = self._build_vi_params()

        # build hidden layers
        for num_units in self._hidden_layers:
            layers.append(
                tfp.layers.DenseVariational(num_units,
                                            variational_distribution,
                                            prior_distribution,
                                            kl_use_exact=self._kl_use_exact,
                                            kl_weight=self._kl_weight,
                                            activation=tf.keras.activations.relu)
            )


        # build output layer

        self._get_output_params()

        layers.append(tfp.layers.DenseVariational(output_dim,
                            variational_distribution,
                            prior_distribution,
                            kl_use_exact=self._kl_use_exact,
                            kl_weight=self._kl_weight,
                            activation=self._output_activation))

        if self._heteroskedastic is True:
            layers.append(tfp.layers.DistributionLambda(
                  lambda t: tfd.Normal(loc=t[..., :self._output_dim],
                                       scale=1e-3+tf.math.softplus(0.05*t[..., self._output_dim:]))))

        elif self._aleatoric_stddev is not None:
                layers.append(tfp.layers.DistributionLambda(lambda t:
                   tfd.Normal(loc=t, scale=self._aleatoric_stddev)))
        else:
            raise Exception("Estimating homoskedastic uncertainty is not "
                            "supported for this model class.")

        return tf.keras.Sequential(layers)

    @check_fitted
    def get_posterior(self, x_test):
        """
        """
        check_array_input(x_test, "x_test")
        raise Exception("Direct estimation of posterior parameters is not "
                        "supported for this model class. Please use the "
                        "sample_posterior() method.")

    @check_fitted
    def sample_posterior(self, x_test):
        """
        """
        check_array_input(x_test, "x_test")
        return self.model(x_test)

    @check_fitted
    @check_aleatoric
    def get_posterior_predictive(self, x_test):
        """
        """
        check_array_input(x_test, "x_test")
        raise Exception("Direct estimation of posterior predictive parameters "
                        "is not supported for this model class. Please use the "
                        "sample_posterior() method.")

    @check_fitted
    @check_aleatoric
    def sample_posterior_predictive(self, x_test):
        """
        """
        check_array_input(x_test, "x_test")
        return self.model(x_test)
