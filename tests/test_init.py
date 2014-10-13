from nose.tools import *
import numpy as np
from matplotlib import pylab

from pysparsa import sparsa


from pykrylov import util as pkutil


def test_simple():
    # simple linear combination of atoms test
    ATOMS = 100
    DIM = 200
    
    A = np.random.normal(0, 1, (DIM, ATOMS))

    x = np.random.rand(ATOMS)
    x[x < 0.8] = 0.0

    y = np.dot(A, x)

    Aop = pkutil.BasicLinOp(A)
    
    res = sparsa.sparsa(y, Aop, 1e-5)
    
    np.testing.assert_array_almost_equal(x, res['x'])
    
