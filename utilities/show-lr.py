import matplotlib.pyplot as plt
from math import cos,pi,e
def lrfn(epoch):
    LR_START = 2.5e-4
    LR_MIN = 1e-5

    lr = cos(epoch*pi/18) * LR_START

    return lr if lr>LR_MIN else LR_MIN

def lrfn1(epoch):
    lr_start = 2.5e-4
    lr_min = 1e-5

    lr = lr_start / (1 + e ** (epoch - 5))
    return lr if lr > lr_min else lr_min

    

epochs = [x for x in range(15)]
lr = [lrfn1(x) for x in epochs]
plt.plot(epochs,lr,'bo')
plt.xlabel('epochs')
plt.ylabel('lr')
for l in lr:
    print(l)
plt.show()

'''
1e-04
0.000166
0.000244
0.000322
0.0002596000000000001
0.00020968000000000004
0.00016974400000000002
0.00013779520000000003
0.00011223616000000004
9.178892800000003e-05
7.543114240000003e-05
6.234491392000002e-05
'''