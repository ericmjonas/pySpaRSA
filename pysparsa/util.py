import numpy as np

def soft(x, T):
    if np.sum(np.abs(T)) == 0:
        y = x
    else:
        y = np.maximum(np.abs(x) - T, 0)
        y = y / (y + T) * x
    return y
