"""
"""

import theano

import logging as logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from kassandra import _base_class
from kassandra.utilities import *

################################################################################

ACTIVATIONS = {
    "relu": theano.tensor.nnet.nnet.relu,
}

################################################################################

class TheanoModel(_base_class.NeuralNetwork):
    """
    """

    def __init__(self, aleatoric=False, heteroskedastic=False,
                 *args, **kwargs):
        """
        """
        super(TheanoModel, self).__init__()

        self._activation = ACTIVATIONS[self._activation]
        self._aleatoric_flag = aleatoric
        self._heteroskedastic = heteroskedastic

        if not isinstance(self._aleatoric_flag, bool):
            self._aleatoric_stddev = math.sqrt(abs(self._aleatoric_flag))
        elif self._aleatoric_flag is False:
            self._aleatoric_stddev = None
        elif self._aleatoric_flag is True and self._heteroskedastic is True:
            self._aleatoric_stddev = None
        else:
            raise ValueError("Invalid argument for aleatoric. Please see "
                             "docstring for valid inputs.")

        if self._regression_flag is False and output_dim == 2:
            raise Exception("For binary classification please set output "
                            "dimensionality to 1.")

        if self._regression_flag is False and self._aleatoric_flag is True:
            raise Exception("Classiffication models automatically estimate "
                            "aleatoric uncertainty. No need to set the "
                            "arguments aleatoric or heteroskedastic.")

        if self._heteroskedastic is True and self._aleatoric_flag is False:
            raise Exception("Setting the aleatoric argument to False assumes "
                            "the homoskedastic regression case. Therefore the "
                            "heteroskedastic argument cannot be set to True.")

        if self._aleatoric_stddev is not None and self._heteroskedastic is True:
            raise Exception("Setting the aleatoric argument to a numeric type"
                            "assumes homoskedastic regression. Therefore the "
                            "heteroskedastic argument cannot be set to True.")

        self._fitted = False

    def save_model(self):
        """
        """
        super().save_model()

    def load_model(self):
        """
        """
        super().load_model()

################################################################################
