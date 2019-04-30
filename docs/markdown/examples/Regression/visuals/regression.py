"""
Modules for GPR visualizations.
"""

import numpy as np
from mpl_toolkits.mplot3d import axes3d
from squidward.utils import exactly_1d

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except:
    import matplotlib
    matplotlib.use('PS')
    from matplotlib import pyplot as plt
    import seaborn as sns

def plot_1d(x_test, mean, var):
    """
    Description
    ----------
    Function to plot one dimensional gaussian process regressor mean and
    variance.

    Parameters
    ----------
    x_test: array_like
        Array containing one dimensional inputs of the gaussian process
        model.
    Mean: array_like
        An array with the values of the mean function of the guassian
        process.
    Var: array_like
        The variance around the values of the mean function of the
        gaussian process.

    Returns
    ----------
    Matplotlib plot of mean function and variance of the gaussian process
    model.
    """
    x_test = exactly_1d(x_test)
    mean = exactly_1d(mean)
    var = exactly_1d(var)

    plt.fill_between(x_test,
                     mean-.674*np.sqrt(var),
                     mean+.674*np.sqrt(var),
                     color='k', alpha=.4, label='50% Credible Interval')
    plt.fill_between(x_test,
                     mean-1.150*np.sqrt(var),
                     mean+1.150*np.sqrt(var),
                     color='k', alpha=.3, label='75% Credible Interval')
    plt.fill_between(x_test,
                     mean-1.96*np.sqrt(var),
                     mean+1.96*np.sqrt(var),
                     color='k', alpha=.2, label='95% Credible Interval')
    plt.fill_between(x_test,
                     mean-2.326*np.sqrt(var),
                     mean+2.326*np.sqrt(var),
                     color='k', alpha=.1, label='99% Credible Interval')
    plt.plot(x_test, mean, c='w')
    return None

def plot_point_grid(model, coordinates=(-1, 1, .1), show_var=False):
    """
    Description
    ----------
    Function to plot a point grid for a two dimensional guassian process
    regressor.

    Parameters
    ----------
    Model: gaussian process regression model object
        A gaussian process regression (gpr) model object.
    Coordinates: Tuple
        A tuple with the minimum and maximum values of the plot grid and
        the interval over which to plot grid points.
        i.e. (min,max,interval)
    Show_var: boolean
        If True, will plot the variance over each point in the grid. If
        False, will plot the mean over each point in the grid.

    Returns
    ----------
    Matplotlib plot of mean function or variance of the gaussian process
    model over each point in the grid.
    """
    x_test, _ = make_grid(coordinates)
    mean, var = model.posterior_predict(x_test)
    mean = exactly_1d(mean)
    var = exactly_1d(var)
    if not show_var:
        plt.scatter(x_test[:, 0], x_test[:, 1], c=mean)
    else:
        plt.scatter(x_test[:, 0], x_test[:, 1], c=var)
    return None

def plot_contour(model, coordinates=(-1, 1, .1), show_var=False):
    """
    Description
    ----------
    Function to plot a contour plot for a two dimensional guassian process
    regressor.

    Parameters
    ----------
    Model: gaussian process regression model object
        A gaussian process regression (gpr) model object.
    Coordinates: Tuple
        A tuple with the minimum and maximum values of the contour and
        the interval over which to the contour.
        i.e. (min,max,interval)
    Show_var: boolean
        If True, will plot the variance contour. If
        False, will plot the mean contour.

    Returns
    ----------
    Matplotlib plot of mean function or variance of the gaussian process
    model as a contour plot.
    """
    x_test, size = make_grid(coordinates)
    mean, var = model.posterior_predict(x_test)
    mean = exactly_1d(mean)
    var = exactly_1d(var)
    if not show_var:
        zed = mean.T.reshape(size, size)
    else:
        zed = np.sqrt(var).T.reshape(size, size)
    alpha, beta = x_test.T.reshape(2, size, size)
    plt.contourf(alpha, beta, zed, 20, cmap='Blues')
    plt.colorbar()
    contours = plt.contour(alpha, beta, zed, 5, colors='black')
    plt.clabel(contours, inline=True, fontsize=8)
    return None

# TODO: Implement
def plot_3d(model, coordinates=(-1, 1, .1), show_var=False):
    """
    Description
    ----------
    Function to make a 3D plot for a two dimensional guassian process
    regressor. Two dimensions are inouts and the third is the target variable.

    Parameters
    ----------
    Model: gaussian process regression model object
        A gaussian process regression (gpr) model object.
    Coordinates: Tuple
        A tuple with the minimum and maximum values of the contour and
        the interval over which to the contour.
        i.e. (min,max,interval)
    Show_var: boolean
        If True, will plot the variance contour. If
        False, will plot the mean contour.

    Returns
    ----------
    Matplotlib plot of mean function or variance of the gaussian process
    model as a 3D plot.
    """
    raise NotImplementedError()
    # x_test, size = make_grid(coordinates)
    # mean, var = model.posterior_predict(x_test)
    # mean = exactly_1d(mean)
    # var = exactly_1d(var)
    # if not show_var:
    #     zed = mean.T.reshape(size, size)
    # else:
    #     zed = np.sqrt(var).T.reshape(size, size)
    # alpha, beta = x_test.T.reshape(2, size, size)
    #
    # fig = plt.figure(figsize=(20, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # #ax = fig.add_subplot(221, projection="3d")
    # ax.plot_surface(alpha, beta, zed, cmap="autumn_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
    # ax.contour(alpha, beta, zed, 10, lw=3, colors="k", linestyles="solid")
    # ax.view_init(30, -60)
    #
    # return None
