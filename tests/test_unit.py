""" This file contains all tests for subfunctions of virga """
import os
import numpy as np


from virga import justdoit as jdi

def test_mie_database():
    qext, qsca, asym, radii, wave = jdi.calc_mie_db(
        ['MnS'], os.path.dirname(__file__), '.', rmin = 1e-5, nradii = 10
    )

    assert np.isclose(np.sum(qext), 3902.080488782566)
    assert np.isclose(np.sum(qsca), 3720.809396082643)
    assert np.isclose(np.sum(asym), 1905.5623111571579)
