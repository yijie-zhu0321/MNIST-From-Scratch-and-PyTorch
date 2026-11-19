import numpy as np


def relu(x):

    return np.maximum(0,x)



def relu_backward(dout,x):

    dx = dout.copy()

    dx[x<=0] = 0

    return dx
