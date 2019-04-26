import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def demo_plot(x_train, y_train, x_test, y_test, x_true, y_true):
    plt.figure(figsize=(10, 4))
    plt.scatter(x_train, y_train, label="train", alpha=0.3)
    plt.scatter(x_test, y_test, label="test", alpha=0.3)
    plt.plot(x_true, y_true, label="True Mean")
    plt.legend()
    plt.show()

def mlp_plots(model, x, x_scatter, y_scatter):

    if model.name != "MLP":
        raise Exception("Not a MLP Model")

    predictions = model.predict(x)

    plt.figure(figsize=(10, 4))
    plt.scatter(x_scatter, y_scatter, label="test")
    plt.scatter(x, predictions, label="pred")
    plt.legend()
    plt.show()

def bn_dropout_plots(model, x, x_scatter, y_scatter):

    if model.name not in ["BNDropout", "BNVI"]:
        raise Exception("Not a BNDropoutModel")

    figsize = (10, 4)
    marker_size = 50

    # Prediction

    plt.figure(figsize=figsize)
    plt.title("Prediction")

    pred = model.predict(x)

    plt.plot(x, pred[0,:], label="Prediction", c="k", lw=3)

    plt.fill_between(x,
             (pred[0,:]-0.674*pred[1,:]),
             (pred[0,:]+0.674*pred[1,:]),
             color="k",alpha=.2,label="50% Credibility Interval"
            )

    plt.fill_between(x,
             (pred[0,:]-1.15*pred[1,:]),
             (pred[0,:]+1.15*pred[1,:]),
             color="k",alpha=.2,label="75% Credibility Interval"
            )

    plt.fill_between(x,
             (pred[0,:]-1.96*pred[1,:]),
             (pred[0,:]+1.96*pred[1,:]),
             color="k",alpha=.2,label="95% Credibility Interval"
            )

    plt.scatter(x_scatter, y_scatter, label="test", marker="x", alpha=1.0, s=marker_size, c="r")
    plt.legend()
    plt.show()

    # Posterior

    plt.figure(figsize=figsize)
    plt.title("Posterior")

    for i in range(500):
        pred = model.sample_posterior(x)
        mu = pred[0, :]
        sigma = pred[1, :]
        plt.plot(x, mu, alpha=0.03, c="k", zorder=-1)

    plt.scatter(x_scatter, y_scatter, label="test", marker="x", alpha=1.0, s=marker_size, c="r", zorder=1)
    plt.legend()
    plt.show()

    # Posterior Predictive

    plt.figure(figsize=figsize)
    plt.title("Posterior Predictive")

    for i in range(10):
        sample = model.sample_posterior_predictive(x)
        plt.scatter(x, sample, alpha=0.05, c="k")

    plt.scatter(x_scatter, y_scatter, label="test", marker="x", alpha=1.0, s=marker_size, c="r")
    plt.legend()
    plt.show()
