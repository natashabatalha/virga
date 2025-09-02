""" This file contains full scale tests of virga """

import numpy as np
import astropy.units as u

from virga import justdoit as jdi

def test_virga_cloud():
    # initialise atmosphere
    a = jdi.Atmosphere(['MnS'], fsed=1, mh=1, mmw=2.2)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())

    # calculate cloud profile
    all_out = jdi.compute(a, as_dict=True, directory='.')

    assert np.isclose(np.sum(all_out['condensate_mmr']), 6.155584963958476e-05)
    
    # initialise atmosphere
    a = jdi.Atmosphere('MnS', fsed=1, mh=1, mmw=2.2)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())

    # calculate cloud profile
    all_out = jdi.compute(a, as_dict=True, directory='.')

    assert np.isclose(np.sum(all_out['condensate_mmr']), 6.155584963958476e-05)

