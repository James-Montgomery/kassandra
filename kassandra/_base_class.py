from abc import ABC, abstractmethod

################################################################################

class BayesianModel(ABC):

    def __init__(self, *args, **kwargs):

        super(BayesianModel, self).__init__(*args, **kwargs)

    @abstractmethod
    def get_prior(self, x_test):
        raise Exception("This method is not supported for this model class.")

    @abstractmethod
    def sample_prior(self, x_test):
        raise Exception("This method is not supported for this model class.")

    @abstractmethod
    def get_prior_predictive(self, x_test):
        raise Exception("This method is not supported for this model class.")

    @abstractmethod
    def sample_prior_predictive(self, x_test):
        raise Exception("This method is not supported for this model class.")

    @abstractmethod
    def fit(self, x_train, y_train):
        raise Exception("This method is not supported for this model class.")

    @abstractmethod
    def get_posterior(self, x_test):
        raise Exception("This method is not supported for this model class.")

    @abstractmethod
    def sample_posterior(self, x_test):
        raise Exception("This method is not supported for this model class.")

    @abstractmethod
    def get_posterior_predictive(self, x_test):
        raise Exception("This method is not supported for this model class.")

    @abstractmethod
    def sample_posterior_predictive(self, x_test):
        raise Exception("This method is not supported for this model class.")

################################################################################

# does this need to be an abstract base class?
class NeuralNetwork(ABC):

    def __init__(self, activation="relu",
                 input_dim=1, hidden_layers=[], output_dim=1,
                 *args, **kwargs):

        if not input_dim >= 1:
            raise ValueError("Input dimensionality must be >= 1.")
        if not output_dim >= 1:
            raise ValueError("Output dimensionality must be >= 1.")

        self._activation = activation
        self._input_dim = input_dim
        self._hidden_layers = hidden_layers
        self._output_dim = output_dim

        super(NeuralNetwork, self).__init__(*args, **kwargs)

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass
