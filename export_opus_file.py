from brukeropusreader import read_file  # noqa: E402

import numpy as np

opusfile = "Lisa_PMIRRAS/2022-11-24_test_hippie.20"  # the full path of the file

Z3 = read_file(opusfile)  # returns a dictionary of the data and metadata extracted

start=Z3["AB Data Parameter"]['FXV']
npts=Z3["AB Data Parameter"]['NPT']
stop=Z3["AB Data Parameter"]['LXV']

X=np.linspace(start,stop,npts)
Y=Z3["AB"]
data=np.stack((X, Y))
print(data.shape)
with open(f'{opusfile}.txt','w') as f:
    np.savetxt(f,data.transpose())