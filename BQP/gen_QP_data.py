import numpy as np
import numpy as np
import scipy.io
import pickle

nsamples = 1000

n = 9
m = 6
nc = 15
mc = 5
# Batch these
c =   2*np.random.rand(nsamples, n);
d =   2*np.random.rand(nsamples, m);
# Static params
roll_new = True
if roll_new:
    Q = np.random.rand(n,n);  Q = Q@Q.T;
    A = np.random.rand(nc,n);
    b = np.random.rand(nc,1)*2*n;
    E = np.random.rand(nc,m);

    H = np.random.rand(m,m);  H = H@H.T;
    e = 2*np.random.rand(m,1);
    f = 2*np.random.rand(n,1);
    F = np.random.rand(mc,m);
    h = np.random.rand(mc,1)*2*m;
    G = np.random.rand(mc,n);
else:
    dict_load = pickle.load(open("./pickle/data_dict_best.p", "rb"))
    Q = dict_load["Q"]
    A = dict_load["A"]
    b = dict_load["b"]
    E = dict_load["E"]

    H = dict_load["H"]
    e = dict_load["e"]
    f = dict_load["f"]
    F = dict_load["F"]
    h = dict_load["h"]
    G = dict_load["G"]

data_dict = dict(c=c,
                 d=d,
                 Q=Q,
                 A=A,
                 b=b,
                 E=E,
                 H=H,
                 e=e,
                 f=f,
                 F=F,
                 h=h,
                 G=G)

scipy.io.savemat('/Users/j/Documents/MATLAB/param_data_{}_{}_{}.mat'.format(n,m,nsamples), data_dict )
pickle.dump( data_dict, open("./pickle/data_dict.p", "wb") )
