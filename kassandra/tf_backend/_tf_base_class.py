"""
"""

import pickle
import math

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from tqdm.keras import TqdmCallback

import logging as logging

tf.get_logger().setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from kassandra import _base_class
from kassandra.utilities import *

################################################################################

ACTIVATIONS = {
    None: None,
    "relu": tf.keras.activations.relu,
}

OPTIMIZERS = {
    "adam" : tf.optimizers.Adam
}

LIKELIHOOD_DISTRIBUTIONS = {
    None: None,
    "guassian": "gaussian",
    "student-t": "student-t",
    "bernoulli": "bernoulli",
    "categorical": "categorical",
}

################################################################################

class TensorflowModel(_base_class.NeuralNetwork):

    def __init__(self, optimizer="adam", lr=0.01, likelihood=None,
                 *args, **kwargs):
        """
        """

        super(TensorflowModel, self).__init__(*args, **kwargs)

        if likelihood is None:
            likelihood = {
                "distribution": "gaussian",
                "parameters": {
                    "mu": None,
                    "std": None
                },
                "heteroskedastic": False
            }

        if likelihood["distribution"] not in LIKELIHOOD_DISTRIBUTIONS.keys():
            raise ValueError("Invalid argument for likelihood distribution.")
        if self._activation not in ACTIVATIONS.keys():
            raise ValueError("Invalid argument for activation function.")
        if optimizer not in OPTIMIZERS.keys():
            raise ValueError("Invalid argument for optimizer.")

        if not isinstance(likelihood["heteroskedastic"], bool):
            raise ValueError("Heteroskedastic flag must be boolean.")
        if likelihood["distribution"] in ["bernoulli", "categorical"] and \
            likelihood["heteroskedastic"] is False:
            aise ValueError("Heteroskedastic flag must be True for"
                            "classification.")

        self._likelihood = likelihood["distribution"]
        self._likelihood_parameters = likelihood["parameters"]
        self._heteroskedastic = likelihood["heteroskedastic"]
        self._activation = ACTIVATIONS[self._activation]
        self._optimizer = OPTIMIZERS[optimizer](learning_rate=lr)

        self._loss = None
        self._fitted = False

    def fit(self, x_train, y_train, epochs=0, verbosity=0):
        """
        """

        check_array_input(x_train, "x_train")
        check_array_input(y_train, "y_train")

        if self._fitted is True:
            logging.warning("Overwriting previously fit parameters.")

        self.model.fit(x_train, y_train,
                       epochs=epochs,
                       verbose=verbosity,
                       callbacks=[TqdmCallback(verbose=0)])

        self._fitted = True

    def save_model(self, file_path="./", model_name="my_model"):
        """
        """

        # save model weights
        self.model.save_weights(file_path + model_name + "/weights")
        #self.weights = self.model.get_weights()

        # clear model from class attributes
        self.model = None

        # save class attributes
        with open(file_path + model_name + "/self.pkl", "wb") as f:
            pickle.dump(self, f)

    def load_model(self, file_path="./", model_name="my_model"):
        """
        """

        # load class attributes
        with open(file_path + model_name + "/self.pkl", "rb") as f:
            self.__dict__.update(pickle.load(f).__dict__)

        # rebuild model
        self.model = self._build()
        self.model.compile(optimizer=self._optimizer, loss=self._loss)

        try:
            # load model weights
            self.model.load_weights(file_path + model_name + "/weights")
            #self.model.set_weights(self.weights)
            #self.weights = None
        except Exception as e:
            # error comnmonly raised on trying to load weights from one model
            # class into another. i.e. MLEMLP into VIMLP
            raise Exception("Error loading model weights. Please make sure \
                that you are trying to load a model of the correct \
                class.\n{}".format(e))

################################################################################
