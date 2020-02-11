try:
    from pkg_resources import declare_namespace
    declare_namespace(__name__)
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)


from . import justdoit 
from . import justplotit

__version__ = 0.0