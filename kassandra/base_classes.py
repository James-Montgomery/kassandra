from abc import ABC

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
# tf.enable_eager_execution()

import numpy as np
import progressbar

"""
args = {"name_1": 1, "name_2" : 2}
Model(*args)
"""


# progressbar object
pbar = progressbar.ProgressBar()


class Model(tf.keras.Model):

    def __init__(self, layers=0, units=1, hidden_activation='relu', output_activation=None, dropout_rate=0,
                 kernel_initializer_magnitude=1.0, bias_initializer_magnitude=1.0, prediction_dropout=False,
                 *args, **kwargs):

        super(self).__init__()

        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0,
                                                                stddev=kernel_initializer_magnitude)

        bias_initializer = tf.keras.initializers.RandomNormal(mean=0.0,
                                                              stddev=bias_initializer_magnitude)

        self.neural_net = []

        for _ in range(layers):

            dense_layer = Dense(units=units,
                                activation=hidden_activation,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer)

            self.neural_net.append(dense_layer)

            if dropout_rate > 0:
                dropout_layer = lambda x: Dropout(rate=dropout_rate,
                                                  noise_shape=(1, units))(x, training=prediction_dropout)

                self.neural_net.append(dropout_layer)

        self.neural_net.append(Dense(units=1,
                                     activation=output_activation,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer))

    def __call__(self, x):

        for layer in self.neural_net:

            x = layer(x)

        return x


class NeuralNetwork(ABC):

    def __init__(self, *args, **kwargs):

        super(self).__init__()

        self.model = Model(output_activation=self.output_activation, *args, **kwargs)

    def fit(self, x_train, y_train, epochs=10, learning_rate=0.01):

        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

        @tf.function
        def train_step(x_train, y_train):
            with tf.GradientTape() as tape:
                predictions = self.model(x_train)

                loss = self.loss_object(y_true=y_train, y_pred=predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            return loss

        losses = []

        for _ in pbar(range(epochs)):

            loss, gradients = train_step(x_train, y_train)
            losses.append(loss.numpy())

        return losses


class NeuralNetworkRegression(ABC, NeuralNetwork):

    def __init__(self):
        self.output_activation = None
        self.loss_object = tf.keras.losses.MeanSquaredError()
        super(self).__init__()
        pass

