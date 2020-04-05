"""
"""

import numpy as np

import logging as logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################################################################

def check_fitted(f):
    """
    """
    def wrapper(*args):
        self = args[0]
        if self._fitted is False:
            logger.warning("This method requries that you first fit your model \
                            to data.")
        return f(*args)
    return wrapper

################################################################################

def check_aleatoric(f):
    """
    """
    def wrapper(*args):
        self = args[0]
        if self._aleatoric_flag is False:
            logger.warning("This method is not supported for models wihtout an \
                            estimation of aleatoric uncertainty.")
        return f(*args)
    return wrapper

################################################################################

def check_array_input(arr, arr_name):
    """
    """

    if not isinstance(arr, np.ndarray):
        raise ValueError("Argument {} must be a numpy array.".format(arr_name))

    if arr.ndim != 2:
        raise ValueError("Argument {} must be a 2D numpy array.".format(arr_name))
