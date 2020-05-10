import matplotlib.pyplot as plt
from math import cos,pi,e
# 来自enet的两个lr调度函数
def lrfn(epoch): #用cos进行退火
    LR_START = 2.5e-4
    LR_MIN = 1e-5

    lr = cos(epoch*pi/18) * LR_START

    return lr if lr>LR_MIN else LR_MIN

def lrfn1(epoch): # 用sigmoid进行学习率退火
    lr_start = 2.5e-4
    lr_min = 1e-5

    lr = lr_start / (1 + e ** (epoch - 5))
    return lr if lr > lr_min else lr_min

    

epochs = [x for x in range(15)]
cos = [lrfn(x) for x in epochs]
sigmoid = [lrfn1(x) for x in epochs]
plt.plot(epochs,cos,'b',label='Cos',color='blue')
plt.plot(epochs,sigmoid,'b',label='Sigmoid',color='green')
plt.xlabel('epochs')
plt.ylabel('lr')
plt.title('Cos V.S. Sigmoid LR Scheduler')
plt.legend()
plt.savefig('lr_scheduler.jpeg')
plt.show()
