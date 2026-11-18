""" Old optics functions """
# pylint: disable=R0914

from .stage_deprecated import sdep_init_optics, sdep_get_refrind

def init_optics(condensibles, nrad=40, rmin=1e-10, read_mie=True):
    """ Deprecated function """
    print(
        '[WARNING] "init_optics" will be fully deprecated in v4. if you are an active '
        'user of this function and there is a rationale we are not aware of for keeping '
        'it please email the developers."'
    )
    return sdep_init_optics(condensibles, nrad=nrad, rmin=rmin, read_mie=read_mie)

def get_refrind(condensible, directory='~/Documents/eddysed/input/optics'):
    """ Deprecated function """
    print(
        '[WARNING] "get_refrind" will be fully deprecated in v4. if you are an active '
        'user of this function and there is a rationale we are not aware of for keeping '
        'it please email the developers."'
    )
    sdep_get_refrind(condensible, directory=directory)
