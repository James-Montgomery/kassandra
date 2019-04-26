import numpy as np

def get_data(N=500, limits=(-10, 10), missing_data=False, heteroscedastic=False):
    l, u = limits

    # features
    x_train = np.random.uniform(l, u, N)
    x_test = np.random.uniform(l, u, N)

    if missing_data:
        # missing data
        x_train = x_train[np.where(np.logical_or(x_train>=-1.0, x_train<=-5.0))]
        x_test = x_test[np.where(np.logical_or(x_test>=-1.0, x_test<=-5.0))]

    if heteroscedastic:
        train_noise = np.random.normal(0, 1.0 + 1.5*np.abs(x_train), x_train.shape[0])
        test_noise = np.random.normal(0, 1.0 + 1.5*np.abs(x_test), x_test.shape[0])
    else:
        train_noise = np.random.normal(0, 3, x_train.shape[0])
        test_noise = np.random.normal(0, 3, x_test.shape[0])

    # targets
    y_train = 1.5*x_train + 5*np.sin(0.8*x_train) + train_noise
    y_test = 1.5*x_test + 5*np.sin(0.8*x_test) + test_noise

    x_true = np.linspace(l, u, 1000)
    y_true = 1.5*x_true + 5*np.sin(0.8*x_true)

    return x_train, x_test, y_train, y_test, x_true, y_true
