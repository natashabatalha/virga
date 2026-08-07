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


    # ==== Test clouds with variable fsed and sigma =========================================
    # initialise atmosphere
    a = jdi.Atmosphere(
        ['MnS', 'SiO2'], fsed={'MnS':1, 'SiO2':1}, mh=1, mmw=2.2, param='exp', b=3,
        sig=[1, 4]
    )
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
        0.007420042794346001, 913.4590758525817, 1144.685910887268, 322829560621.1157,
        3400236589.2062654, 36062700139.00066
    ]
    for i, test in enumerate(tested_outputs):
        assert np.isclose(np.sum(all_out[test]), expected_outputs[i])


def test_virtual_cloud_and_og_vfall():
    # ==== Basic run ====================================================================
    # initialise atmosphere
    a = jdi.Atmosphere(['MnS'], fsed=1, mh=1, mmw=2.2)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    test = jdi.hot_jupiter()
    test['temperature'] /= 100
    a.ptk(df=test)
    # calculate cloud profile
    all_out = jdi.compute(
        a, as_dict=True, directory=os.path.dirname(__file__), og_vfall=False
    )
    # test the output
    tested_outputs = [
        'condensate_mmr', 'mean_particle_r', 'droplet_eff_r', 'column_density',
        'single_scattering', 'asymmetry', 'opd_by_gas', 'mixing_length', 'altitude',
    ]
    expected_outputs = [
        6.21733257330643e-05, 1320688.476214182, 4389808.9567320105,
        0.16073295909273821, 9633.44498928589, 6579.989951588848, 1.0269292661864409,
        34002365.89206266, 360627001.39000654
    ]
    for i, test in enumerate(tested_outputs):
        assert np.isclose(np.sum(all_out[test]), expected_outputs[i])


def test_direct_solver():
    # ==== Basic run ====================================================================
    # initialise atmosphere
    a = jdi.Atmosphere(['MnS'], fsed=1, mh=1, mmw=2.2)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    # calculate cloud profile
    all_out = jdi.compute(a, as_dict=True, directory=os.path.dirname(__file__),
                          og_solver=False, og_vfall=False)
    # test the output
    tested_outputs = [
        'condensate_mmr', 'mean_particle_r', 'droplet_eff_r', 'column_density',
        'single_scattering', 'asymmetry', 'opd_by_gas', 'mixing_length', 'altitude',
    ]
    expected_outputs = [
        1.843949389772658e-05, 732.9302063055602, 2436.1714683257405,
        56.98667503728258, 5958.077421399466, 3776.112042191915, 0.2717684171959098,
        681615970.0850405, 35075608885.320946
    ]
    for i, test in enumerate(tested_outputs):
        assert np.isclose(np.sum(all_out[test]), expected_outputs[i])

def test_gamma():
    # ==== Basic run ====================================================================
    # initialise atmosphere
    a = jdi.Atmosphere(['MnS'], fsed=1, mh=1, mmw=2.2, dist='gamma')
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
        6.163946287910695e-05, 514.7395678038475, 919.6026299769981, 981.9688907809427,
        5903.838459340235, 3255.8207782168574, 0.6933963285465201, 3400236589.2062654,
        36062700139.00066
    ]
    for i, test in enumerate(tested_outputs):
        assert np.isclose(np.sum(all_out[test]), expected_outputs[i])

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
    all_out = jdi.compute(
        a, as_dict=True, directory=os.path.dirname(__file__), mixed_opacity_type='quick'
    )
    # test the output
    tested_outputs = [
        'condensate_mmr', 'mean_particle_r', 'droplet_eff_r', 'column_density',
        'mixing_length', 'altitude',
    ]
    expected_outputs = [
        0.004797725874827489, 457.4281311886753, 1520.435851086795, 4349822.742474344,
        3400236589.2062654, 36062700139.00066
    ]

    for i, test in enumerate(tested_outputs):
        assert np.isclose(np.sum(all_out[test]), expected_outputs[i])

    # ==== Test mixed clouds with variable fsed =========================================
    # initialise atmosphere
    a = jdi.Atmosphere(
        ['MnS', 'SiO2'], fsed={'MnS':1, 'SiO2':2, 'mixed': 3}, mh=1, mmw=2.2, mixed=True
    )
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    # calculate cloud profile
    all_out = jdi.compute(
        a, as_dict=True, directory=os.path.dirname(__file__), mixed_opacity_type='quick'
    )
    # test the output
    tested_outputs = [
        'condensate_mmr', 'mean_particle_r', 'droplet_eff_r', 'column_density',
        'mixing_length', 'altitude',
    ]
    expected_outputs = [
        0.0024403823704852393, 943.2559217922122, 3135.2687394104623, 501618.68059201626,
        3400236589.2062654, 36062700139.00066
    ]
    for i, test in enumerate(tested_outputs):
        assert np.isclose(np.sum(all_out[test]), expected_outputs[i])

    # ==== Test mixed clouds with variable fsed =========================================
    # initialise atmosphere
    a = jdi.Atmosphere(['MnS', 'SiO2'], fsed=1, mh=1, mmw=2.2, mixed=False)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    # calculate cloud profile
    all_out = jdi.compute(a, as_dict=True, directory=os.path.dirname(__file__),
                          mixed_opacity_type='quick')
    # test the output
    tested_outputs = [
        'condensate_mmr', 'mean_particle_r', 'droplet_eff_r', 'column_density',
        'mixing_length', 'altitude',
    ]
    expected_outputs = [
        0.00479764116888752, 674.8984205851, 2243.280822788288, 4350662.838922345,
        3400236589.2062654, 36062700139.00066
    ]
    for i, test in enumerate(tested_outputs):
        assert np.isclose(np.sum(all_out[test]), expected_outputs[i])
