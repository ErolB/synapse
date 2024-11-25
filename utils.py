import pandas as pd
import numpy as np
import random

# activation functions
def sigmoid(z):
    return np.exp(z) / (1 + np.exp(z))

def relu(z):
    if z < 0:
        return 0
    else:
        return z

def linear(z):
    return z

# graph functions

