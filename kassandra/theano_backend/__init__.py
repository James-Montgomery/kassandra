try:
    # for local testing
    from . import models
except:
    # when installed
    from kassandra.theano_backend import models
