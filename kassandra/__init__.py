"""
Kassandra
=====
A package for simplifying bayesian deep learning.

How to use the documentation
----------------------------
The docstring examples assume that `kassandra` has been imported as `kass`::
  >>> import kassandra as kass
Code snippets are indicated by three greater-than signs::
  >>> x = 42
  >>> x = x + 1
Use the built-in ``help`` function to view a function's docstring::
  >>> help(kass)
 or ``dir``::
  >>> dir(kass)
You can check the annotations of a function to see type hints::
  >>> kass.models.MLEMLP().fit.__annotations__
By conventions, look to module docstrings for source references and definitions
of important terms / acronyms.
  >>> help(kass.models)

Available subpackages
---------------------
models
    the module containing the model classes for deep learning
"""

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# TODO: docstrings, type hinting, pylint, pytest, docs

try:
    # for local testing
    from . import models
except:
    # when installed
    import kassandra.models
