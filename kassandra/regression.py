from kassandra._base_regression_models import _DropoutModel
from kassandra._base_classes import _FrequentistModel, _BayesianModel

import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------------------------------------
# Neural Network: Multi-Layer Perceptron
# ---------------------------------------------------------------------------------------------------------------------


class MLP(_FrequentistModel):

    def __init__(self, aleatoric_uncertainty=None, *args, **kwargs):

        super(_FrequentistModel, self).__init__()
        self.name = "Neural Network: Multi-Layer Perceptron"

        self._model = _DropoutModel(mc_dropout=False,
                                    aleatoric_uncertainty=aleatoric_uncertainty,
                                    *args, **kwargs)

        self.loss_object = self._model.get_loss_object()

# ---------------------------------------------------------------------------------------------------------------------
# Bayesian Neural Network: MC Dropout
# ---------------------------------------------------------------------------------------------------------------------


class BNDropout(_BayesianModel):

    def __init__(self, aleatoric_uncertainty="homoscedastic", *args, **kwargs):

        super(_BayesianModel, self).__init__()
        self.name = "Bayesian Neural Network: MC Dropout"

        self._model = _DropoutModel(mc_dropout=True,
                                    aleatoric_uncertainty=aleatoric_uncertainty,
                                    *args, **kwargs)

        self.loss_object = self._model.get_loss_object()

    def get_prior(self, *args, **kwargs):
        raise Exception("Prior methods not supported for MC Dropout approximate inference method.")

    def get_prior_predictive(self, *args, **kwargs):
        raise Exception("Prior methods not supported for MC Dropout approximate inference method.")

    def sample_prior(self, *args, **kwargs):
        raise Exception("Prior methods not supported for MC Dropout approximate inference method.")

    def sample_prior_predictive(self, *args, **kwargs):
        raise Exception("Prior methods not supported for MC Dropout approximate inference method.")

# ---------------------------------------------------------------------------------------------------------------------
# Bayesian Neural Network: Variational Inference
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# Bayesian Neural Network: Markov Chain Monte Carlo
# ---------------------------------------------------------------------------------------------------------------------
