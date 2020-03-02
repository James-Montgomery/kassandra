try:
    # for local testing
    from . import models
except:
    # when installed
    import kassandra.theano_backend.models
