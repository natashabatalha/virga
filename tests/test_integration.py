""" This file contains full scale tests of virga """
import os
import numpy as np
import astropy.units as u
import pandas as pd

from virga import justdoit as jdi

def test_basic_virga():
    # ==== Basic run ====================================================================
    # initialise atmosphere
    a = jdi.Atmosphere(['MnS'], fsed=1, mh=1, mmw=2.2)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    # calculate cloud profile
    all_out = jdi.compute(a, as_dict=True, directory=os.path.dirname(__file__))
    # test the output
    tested_outputs = [
        'condensate_mmr', 'mean_particle_r', 'droplet_eff_r', 'column_density',
        'single_scattering', 'asymmetry', 'opd_by_gas', 'mixing_length', 'altitude',
    ]
    expected_outputs = [
        6.155584963958476e-05, 203.8323901780218, 677.5142421477923,
        4277.703510668625, 5919.737864805443, 3171.822432592176, 0.960369799652179,
        3400236589.2062654, 36062700139.00066,
    ]
    for i, test in enumerate(tested_outputs):
        assert np.isclose(np.sum(all_out[test]), expected_outputs[i])

    # ==== additional unit tests done here because we already run virga =================
    df_cl = jdi.picaso_format(
        all_out['opd_per_layer'], all_out['single_scattering'], all_out['asymmetry'],
        pressure=all_out['pressure'], wavenumber=1/all_out['wave']/1e-4,
    )
    df_cl.to_csv(os.path.dirname(__file__) + '/picaso_format_test.csv', index=False)
    df_test = pd.read_csv(os.path.dirname(__file__) + '/picaso_format_test.csv')
    pd.testing.assert_frame_equal(df_cl, df_test)

    # ==== Test mixed clouds with variable fsed =========================================
    # initialise atmosphere
    a = jdi.Atmosphere(['MnS', 'SiO2'], fsed={'MnS':1, 'SiO2':1}, mh=1, mmw=2.2, param='exp', b=3)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    # calculate cloud profile
    all_out = jdi.compute(a, as_dict=True, directory=os.path.dirname(__file__))
    # test the output
    tested_outputs = [
        'condensate_mmr', 'mean_particle_r', 'droplet_eff_r', 'column_density',
        'mixing_length', 'altitude',
    ]
    expected_outputs = [
        0.007420224549370982, 385.37988982782247, 1280.956200178287, 94307.79521654578,
        3400236589.2062654, 36062700139.00066,
    ]
    for i, test in enumerate(tested_outputs):
        assert np.isclose(np.sum(all_out[test]), expected_outputs[i])

    # ==== testing direct solver, analytic radius calc, and original fall velocity calc
    # initialise atmosphere
    a = jdi.Atmosphere('MnS', fsed=1, mh=1, mmw=2.2)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    # calculate cloud profile
    all_out = jdi.compute(a, as_dict=True, directory=os.path.dirname(__file__),
                          og_solver=False, analytical_rg=False, og_vfall=False, do_virtual=False)
    assert np.isclose(np.sum(all_out['condensate_mmr']), 1.843949389772658e-05)

def test_fractals():
    # initialise atmosphere
    a = jdi.Atmosphere(['MnS'], fsed=1, mh=1, mmw=2.2, aggregates=True, Df=2, N_mon=100)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    # calculate cloud profile
    all_out = jdi.compute(a, as_dict=True, directory=os.path.dirname(__file__))
    # test the output
    tested_outputs = [
        'condensate_mmr', 'mean_particle_r', 'droplet_eff_r', 'column_density',
        'single_scattering', 'asymmetry', 'opd_by_gas', 'mixing_length', 'altitude',
    ]
    expected_outputs = [
        6.163940981671749e-05, 455.73378048873894, 1514.8040340365344,
        172.35717609739288, 4956.102590299978, 221.52557712010397, 0.5743750726911432,
        3400236589.2062654, 36062700139.00066,
    ]
    for i, test in enumerate(tested_outputs):
        assert np.isclose(np.sum(all_out[test]), expected_outputs[i])

def test_mixed_clouds():
    # ==== Test normal mixed clouds =====================================================
    # initialise atmosphere
    a = jdi.Atmosphere(['MnS', 'SiO2'], fsed=1, mh=1, mmw=2.2, mixed=True)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    # calculate cloud profile
    all_out = jdi.compute(a, as_dict=True, directory=os.path.dirname(__file__), quick_mix=True)
    # test the output
    tested_outputs = [
        'condensate_mmr', 'mean_particle_r', 'droplet_eff_r', 'column_density', 'mixing_length',
        'altitude',
    ]
    expected_outputs = [
        0.009600444481429805, 741.1002992334354, 2463.327870276736, 23435.298259890198,
        3400236589.2062654, 36062700139.00066,
    ]

    for i, test in enumerate(tested_outputs):
        assert np.isclose(np.sum(all_out[test]), expected_outputs[i])

    # ==== Test mixed clouds with variable fsed =========================================
    # initialise atmosphere
    a = jdi.Atmosphere(['MnS', 'SiO2'], fsed={'MnS':1, 'SiO2':2, 'mixed': 3}, mh=1, mmw=2.2, mixed=True)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    # calculate cloud profile
    all_out = jdi.compute(a, as_dict=True, directory=os.path.dirname(__file__), quick_mix=True)
    # test the output
    tested_outputs = [
        'condensate_mmr', 'mean_particle_r', 'droplet_eff_r', 'column_density',
        'mixing_length', 'altitude',
    ]
    expected_outputs = [
        0.004892882346521104, 1260.8837468647716, 4191.0252607422235, 4541.284526267978,
        3400236589.2062654, 36062700139.00066,
    ]
    for i, test in enumerate(tested_outputs):
        assert np.isclose(np.sum(all_out[test]), expected_outputs[i])

    # ==== Test mixed clouds with variable fsed =========================================
    # initialise atmosphere
    a = jdi.Atmosphere(['MnS', 'SiO2'], fsed=1, mh=1, mmw=2.2, mixed=False)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    # calculate cloud profile
    all_out = jdi.compute(a, as_dict=True, directory=os.path.dirname(__file__))
    # test the output
    tested_outputs = [
        'condensate_mmr', 'mean_particle_r', 'droplet_eff_r', 'column_density',
        'mixing_length', 'altitude',
    ]
    expected_outputs = [
        0.004800222240714903, 479.5232340104981, 1593.8773037942406, 12144.145061051053,
        3400236589.2062654, 36062700139.00066,
    ]
    for i, test in enumerate(tested_outputs):
        assert np.isclose(np.sum(all_out[test]), expected_outputs[i])
