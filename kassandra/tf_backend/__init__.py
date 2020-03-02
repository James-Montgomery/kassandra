try:
    # for local testing
    from . import models
except:
    # when installed
    import kassandra.tf_backend.models
