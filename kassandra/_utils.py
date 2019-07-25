import random
import time

import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def override(f):
    """

    """
    return f


def shuffle(x, y, N):
    """

    """
    ind_list = [i for i in range(N)]
    random.shuffle(ind_list)
    x = x[ind_list]
    y = y[ind_list]
    return x, y


class Timer:
    """

    """
    def __init__(self):
        """

        """
        pass

    def __enter__(self):
        """

        """
        self.t0 = time.time()

    def __exit__(self, *args):
        """

        """
        print('Elapsed time: %0.2fs' % (time.time()-self.t0))
