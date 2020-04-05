import pymc3 as pm

import logging as logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from kassandra import _base_class
from kassandra.theano_backend import _theano_base_class
from kassandra.utilities import *

################################################################################

VARIATIONAL_DISTRIBUTIONS = {
    "independant_gaussians": "independant_gaussians",
    "mean_field_approximation": "independant_gaussians",
}
PRIOR_DISTRIBUTIONS = VARIATIONAL_DISTRIBUTIONS

DISTRIBUTION_ATTRIBUTES = {
    "independant_gaussians": ["mu", "std"],
    "mean_field_approximation": ["mu", "std"]
}

################################################################################
#
# class VIMLP(_theano_base_class.TheanoModel, _base_class.BayesianModel):
#     """
#     Variational Inference for Multi-Layer Perceptron
#     """
#
#     def __init__(self,
#                  prior_distribution="independant_gaussians",
#                  prior_parameters={"mu": 0.0, "std": 1.0},
#                  *args, **kwargs):
#         """
#         """
#
#         super(VIMLP, self).__init__(*args, **kwargs)
#
#         self._variational_distribution = \
#             VARIATIONAL_DISTRIBUTIONS[variational_distribution]
#         self._prior_distribution = PRIOR_DISTRIBUTIONS[prior_distribution]
#         self._prior_parameters = prior_parameters
#
#
#     def fit(self, x_train, y_train):
#         """
#         """
#
#         check_array_input(x_train, "x_train")
#         check_array_input(y_train, "y_train")
#
#         ann_input = theano.shared(x_train)
#         ann_output = theano.shared(y_train)
#
#         if self._fitted is True:
#             logging.warning("Overwriting previously fit parameters.")
#
#         # dtype = theano.config.floatX
#         # Initialize random weights between each layer
#         # init_w_0 = np.random.randn(X.shape[1], n_hidden).astype(dtype)
#         # init_b_0 = np.random.randn(n_hidden).astype(dtype)
#
#         weights = []
#         biases = []
#
#         with pm.Model() as model:
#
#             if self._aleatoric_stddev
#
#             if len(layers) == 0:
#                 # No hidden layers
#
#                 w = pm.Normal('w_0', 0, sd=1, shape=(x_train.shape[1], self._output_dim))#, testval=init_1)
#                 b = pm.Normal('b_0', 0, sd=1, shape=(self._output_dim))#, testval=init_b_1)
#
#                 variance = pm.HalfNormal('uncertainty', sigma=1.0)
#                 out = pm.Normal('out', mu=act_out, sigma=variance, observed=ann_output)
#
#             else:
#                 # Hidden layers
#
#                 # Weights from input to hidden layer
#                 w = pm.Normal('w_0', 0, sd=1, shape=(x_train.shape[1], self._hidden_layers[0]))
#                 b = pm.Normal('b_0', 0, sd=1, shape=(self._hidden_layers[0]))
#
#                 for i, layer in enumerate(self._hidden_layers):
#
#                     w = pm.Normal('w_0', 0, sd=1, shape=(x_train.shape[1], n_hidden))
#                     b = pm.Normal('b_0', 0, sd=1, shape=(n_hidden))
#
#
#                 # Build neural-network using tanh activation function
#                 act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1) + weights_b_1)
#                 act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2) + weights_b_2)
#                 act_out = pm.math.dot(act_2, weights_2_out) + weights_b_out
#
#                 variance = pm.HalfNormal('uncertainty', sigma=1.0)
#                 out = pm.Normal('out', mu=act_out, sigma=variance, observed=ann_output)
#
#
#
#
#
#         self.model = model
#         self._fitted = True
