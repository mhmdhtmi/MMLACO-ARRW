import numpy as np, pandas as pd, sys
sys.path.insert(0,"src")
from mmlaco_arrw import alpha_gamma, compute_mutual_information, mmlaco
def test_arrw_endpoints():
    a0,g0=alpha_gamma(0,50,mode="arrw"); a1,g1=alpha_gamma(49,50,mode="arrw")
    assert np.isclose(a0,.9) and np.isclose(g0,.1) and np.isclose(a1,.1) and np.isclose(g1,.9)
def test_fixed_weights():
    for i in [0,1,49]:
        a,g=alpha_gamma(i,50,mode="fixed"); assert np.isclose(a,.9) and np.isclose(g,.1)
def test_seed_reproducibility():
    rng=np.random.default_rng(11); data=pd.DataFrame(rng.integers(0,3,(30,8))); labels=pd.DataFrame(rng.integers(0,2,(30,3)))
    mif,mil=compute_mutual_information(data,labels)
    a,ta,ha=mmlaco(data,mif,mil,n_iter=5,n_ants=4,n_selected=3,random_state=123)
    b,tb,hb=mmlaco(data,mif,mil,n_iter=5,n_ants=4,n_selected=3,random_state=123)
    assert np.array_equal(a,b) and np.allclose(ta,tb) and ha.equals(hb)
