""" This file contains all tests for subfunctions of virga """
import os
import numpy as np
from astropy.units import temperature

from virga import justdoit as jdi

def test_mie_database():

    # ==== Fortran (this one thakes way to long to calcualte, but it works)
    # qext, qsca, asym, radii, wave = jdi.calc_mie_db(
    #     ['MnS'], os.path.dirname(__file__), '.', rmin = 1e-5, nradii = 10, fort_calc_mie=True
    # )
    #
    # assert np.isclose(np.sum(qext), 3902.080488782566)
    # assert np.isclose(np.sum(qsca), 3720.809396082643)
    # assert np.isclose(np.sum(asym), 1905.5623111571579)

    # # ==== Fractals
    # qext, qsca, asym, radii, wave = jdi.calc_mie_db(
    #     ['MnS'], os.path.dirname(__file__), '.', rmin = 1e-5, nradii = 10,
    #     aggregates=os.path.dirname(__file__), Df=1, N_mon=100,
    #     optool_dir='/home/kiefersv/Documents/work/not_my_code/optool'
    # )
    # print(np.sum(qext))
    # print(np.sum(qsca))
    # print(np.sum(asym))
    # assert np.isclose(np.sum(qext), 3902.080488782566)
    # assert np.isclose(np.sum(qsca), 3720.809396082643)
    # assert np.isclose(np.sum(asym), 1905.5623111571579)

    # ==== Basic
    qext, qsca, asym, radii, wave = jdi.calc_mie_db(
        ['MnS'], os.path.dirname(__file__), '.', rmin = 1e-5, nradii = 10
    )

    assert np.isclose(np.sum(qext), 3902.080488782566)
    assert np.isclose(np.sum(qsca), 3720.809396082643)
    assert np.isclose(np.sum(asym), 1905.5623111571579)

# def test_gas_properties():
#     from virga.gas_properties import (
#         TiO2, CH4, NH3, H2O, Fe, KCl, MgSiO3, Mg2SiO4, MnS, ZnS, Cr, Al2O3, Na2S,
#         CaTiO3, CaAl12O19, SiO2
#     )
#     # common setup
#     mh = 2
#     mmw = 2.34
#     functions = [
#         TiO2, CH4, NH3, H2O, Fe, KCl, MgSiO3, Mg2SiO4, MnS, ZnS, Cr, Al2O3, Na2S,
#         CaTiO3, CaAl12O19, SiO2
#     ]
#     results = [
#         84.25001155555556, 16.496700854700855, 17.84194700854701, 18.9416,
#         63.722431882692305, 76.49001623717949, 103.59450570940172, 143.91768345384614,
#         91.00004022820512, 101.50000633073505, 59.14603941918975, 105.94821690677692,
#         79.9062228094017, 139.73229121363246, 674.9935023092648, 62.65109230769231
#     ]
#     # test each material
#     for f, func in enumerate(functions):
#         assert np.isclose(np.sum(func(mmw, mh)), results[f])
#
# def test_pvaps():
#     from virga.pvaps import (
#         TiO2, CH4, NH3, H2O, Fe, KCl, MgSiO3, Mg2SiO4, MnS, ZnS, Cr, Al2O3, Na2S,
#         CaTiO3, CaAl12O19, SiO2
#     )
#     # common setup
#     temps = np.linspace(200, 2000, 200)
#     pressure = 1
#     functions_wo_p = [TiO2, CH4, NH3, H2O, Fe, KCl, MgSiO3, MnS, ZnS, Cr, Al2O3, Na2S, SiO2]
#     functions_w_p = [Mg2SiO4, CaTiO3, CaAl12O19]
#     results_wo_p = [
#         34.91961102992913, 60753502595926.91, 2819322601934.0454, 83261130546.28918,
#         3514.1217876496853, 1268153768.5145688, 856084.7997978945, 3404023.8269615686,
#         859913289854.2339, 14278.409833990157, 15.683552452135084, 518386955.4727935,
#         751356.190783691,
#     ]
#     results_w_p = [4203759.617075334, 4525511.860469028, 41.81826235227096]
#     # test each material
#     for f, func in enumerate(functions_wo_p):
#         result = 0
#         for temp in temps:
#             result += func(temp)
#         assert np.isclose(result, results_wo_p[f])
#     for f, func in enumerate(functions_w_p):
#         result = 0
#         for temp in temps:
#             result += func(temp, pressure)
#         assert np.isclose(result, results_w_p[f])


