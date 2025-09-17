""" This file contains full scale tests of virga """
import os
import numpy as np
import astropy.units as u

from virga import justdoit as jdi

def test_basic_virga():
    # ==== Basic run
    # initialise atmosphere
    a = jdi.Atmosphere(['MnS'], fsed=1, mh=1, mmw=2.2)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    # calculate cloud profile
    all_out = jdi.compute(a, as_dict=True, directory=os.path.dirname(__file__))
    assert np.isclose(np.sum(all_out['condensate_mmr']), 6.155584963958476e-05)

    # ==== Basic run, with string input
    # initialise atmosphere
    a = jdi.Atmosphere('MnS', fsed=1, mh=1, mmw=2.2)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    # calculate cloud profile
    all_out = jdi.compute(a, as_dict=True, directory=os.path.dirname(__file__))
    assert np.isclose(np.sum(all_out['condensate_mmr']), 6.155584963958476e-05)

    # ==== testing direct solver
    # initialise atmosphere
    a = jdi.Atmosphere('MnS', fsed=1, mh=1, mmw=2.2)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    # calculate cloud profile
    all_out = jdi.compute(a, as_dict=True, directory=os.path.dirname(__file__),
                          og_solver=False)
    assert np.isclose(np.sum(all_out['condensate_mmr']), 1.843949389772658e-05)

def test_fractals():
    # initialise atmosphere
    a = jdi.Atmosphere(['MnS'], fsed=1, mh=1, mmw=2.2, aggregates=True, Df=2, N_mon=100)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())

    # calculate cloud profile
    all_out = jdi.compute(a, as_dict=True, directory=os.path.dirname(__file__))

    assert np.isclose(np.sum(all_out['condensate_mmr']), 6.163940981671749e-05)
