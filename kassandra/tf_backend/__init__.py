try:
    # for local testing
    from . import models
except:
    # when installed
    from kassandra.tf_backend import models
