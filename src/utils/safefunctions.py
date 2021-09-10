import numpy as np

def safesqrt(x):
    return np.sqrt(np.abs(x))

def safediv(x, y):
    safe_y = np.where(abs(y) < 0.00001, abs(y) + 0.00001, y)
    return x / safe_y

def safepow(x, y):
    return np.power(np.where(np.abs(x)+0.0001>10, 10, np.abs(x)+0.00001), np.where(abs(y) > 10, 10, y))