from abc import ABC, abstractmethod
import pickle
import math


from tqdm.keras import TqdmCallback

import tensorflow as tf

import logging

tf.get_logger().setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################################################################

ACTIVATIONS = {
    "relu": tf.keras.activations.relu,
}

OPTIMIZERS = {
    "adam" : tf.optimizers.Adam
}


tf.optimizers.Adam(learning_rate=0.05)

################################################################################

class NeuralNetwork(ABC):

    def __init__(self, activation="relu", num_hidden_layers=0, num_hidden_units=1,
                 optimizer="adam", lr=0.01, regression=True, output_dim=1,
                 aleatoric=False, heteroskedastic=False,
                 known_aleatoric_variance=None, *args, **kwargs):

        self._activation = ACTIVATIONS[activation]
        self._num_hidden_layers = num_hidden_layers
        self._num_hidden_units = num_hidden_units
        self._optimizer = OPTIMIZERS[optimizer](learning_rate=lr)
        self._regression_flag = regression
        self._output_dim = output_dim
        self._aleatoric_flag = aleatoric
        self._heteroskedastic = heteroskedastic
        self._aleatoric_stddev = None if known_aleatoric_variance is None else \
            math.sqrt(abs(known_aleatoric_variance))

        self._loss = None
        self._fitted = False

        if self._regression_flag is False and output_dim == 2:
            raise Exception("For binary classification please use output \
                dimension 1.")

        if self._regression_flag is False and self._aleatoric_flag is True:
            raise Exception("Classiffication models automatically estimate \
                aleatoric uncertainty. No need to set the arguments \
                aleatoric, heteroskedastic, or known_aleatoric_variance")

        if self._aleatoric_stddev is not None and self._aleatoric_flag is False:
            raise Exception("The aleatoric argument must be set to True in \
                order to use the known_aleatoric_variance argument.")

        if self._heteroskedastic is True and self._aleatoric_flag is False:
            raise Exception("Setting the aleatoric argument to False assumes \
                homoskedastic regression. Therefore the heteroskedastic \
                argument cannot be set to True.")

        if self._aleatoric_stddev is not None and self._heteroskedastic is True:
            raise Exception("Setting the known_aleatoric_variance argument \
                assumes homoskedastic regression. Therefore the heteroskedastic \
                argument cannot be set to True.")

        super().__init__()

    def fit(self, x_train, y_train, epochs=0, verbosity=0):

        self.model.fit(x_train, y_train,
                       epochs=epochs,
                       verbose=verbosity,
                       callbacks=[TqdmCallback(verbose=0)])

        self._fitted = True

    def save(self, file_path="./", model_name="my_model"):

        # save model weights
        self.model.save_weights(file_path + model_name + "/weights")
        #self.weights = self.model.get_weights()

        # clear model from class attributes
        self.model = None

        # save class attributes
        with open(file_path + model_name + "/self.pkl", "wb") as f:
            pickle.dump(self, f)

    def load(self, file_path="./", model_name="my_model"):

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
