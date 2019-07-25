import tensorflow as tf
import progressbar

from abc import ABC, abstractmethod
import numpy as np

import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------------------------------------
# Neural Net Base Class
# ---------------------------------------------------------------------------------------------------------------------


class _NeuralNetwork(ABC):

    def __init__(self):
        super(_NeuralNetwork, self).__init__()
        self._fitted = False

    def fit(self, x_train, y_train, epochs=10, learning_rate=0.01):

        loss_object = self.loss_object
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

        @tf.function  # comment out decorator for easier debugging
        def train_step(x_train, y_train):

            with tf.GradientTape() as tape:

                if self._model.aleatoric_uncertainty is None:
                    predictions = self._model(x_train)
                    loss = loss_object(y_true=y_train, y_pred=predictions)
                else:
                    mean, sigma = self._model(x_train)
                    loss = loss_object(mean, sigma, y_train)

            gradients = tape.gradient(loss, self._model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

            return loss

        losses = []
        pbar = progressbar.ProgressBar()
        for _ in pbar(range(epochs)):

            step_loss = train_step(x_train, y_train)

            losses.append(step_loss.numpy())

        self._fitted = True
        return losses

# ---------------------------------------------------------------------------------------------------------------------
# Frequentist Model Base Class
# ---------------------------------------------------------------------------------------------------------------------


class _FrequentistModel(_NeuralNetwork):

    def __init__(self):
        super(_NeuralNetwork, self).__init__()

    def predict(self, x_test):

        if not self._fitted:
            logger.warning("Making predictions with unfit model.")

        if self._model.aleatoric_uncertainty is None:
            return self._model(x_test).numpy()

        else:
            mean, sigma = self._model(x_test)
            return mean.numpy(), sigma.numpy()

# ---------------------------------------------------------------------------------------------------------------------
# Bayesian Model Base Class
# ---------------------------------------------------------------------------------------------------------------------


def posterior_method(func):
    def wrapper(self, *args, **kwargs):
        assert self._fitted, "Please fit model before calling posterior methods."
        return func(self, *args, **kwargs)
    return wrapper


def predictive_method(func):
    def wrapper(self, *args, **kwargs):
        assert self._model.aleatoric_uncertainty is not None, "Predictive methods not supported if " \
                                                       "aleatoric_uncertainty is None"
        return func(self, *args, **kwargs)
    return wrapper


class _BayesianModel(_NeuralNetwork):

    def __init__(self):
        super(_NeuralNetwork, self).__init__()

    def _predict(self, x_test):
        if self._model.aleatoric_uncertainty is None:
            return self._model(x_test).numpy()
        mean, simga = self._model(x_test).numpy()
        return mean

    def get_prior(self):
        pass

    @predictive_method
    def get_prior_predictive(self):
        pass

    @posterior_method
    def get_posterior(self, x_test, n_estimates=10, return_cov=False):
        assert n_estimates >= 1, "Invalid argument for n_estimators. Must be integer >= 1"

        samples = np.asarray([
            self._predict(x_test)
            for _ in range(n_estimates)
        ]).squeeze()

        mean = samples.mean(axis=0)
        var = samples.std(axis=0) ** 2

        if not return_cov:
            return mean, var

        cov = np.cov(samples.T)
        return mean, cov

    @posterior_method
    @predictive_method
    def get_posterior_predictive(self):
        pass

    def sample_prior(self):
        pass

    @predictive_method
    def sample_prior_predictive(self):
        pass

    @posterior_method
    def sample_posterior(self):
        pass

    @posterior_method
    @predictive_method
    def sample_posterior_predictive(self):
        pass

