from brukeropusreader import read_file  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np

opusfile = "Lisa_PMIRRAS/2022-11-24_test_hippie.7"  # the full path of the file

Z3 = read_file(opusfile)  # returns a dictionary of the data and metadata extracted
for key in Z3:
    # print(key)
    pass
    # print(Z3[key])

print(Z3["AB"].shape)  # looks what is the Optik block:

### need to build X axis ##
# print(Z3["Fourier Transformation"])

start=Z3["AB Data Parameter"]['FXV']
npts=Z3["AB Data Parameter"]['NPT']
stop=Z3["AB Data Parameter"]['LXV']

X=np.linspace(start,stop,npts)
Y=Z3["AB"]

plt.plot(X,Y)
plt.show()
