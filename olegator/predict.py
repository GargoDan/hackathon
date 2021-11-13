import pandas as pd
import numpy as np
from scipy.stats import randint, norm

def make_bootstrap(X):
    Y = randint.rvs(low = 0, high = len(X),  size = len(X))
    
    return [X[i] for i in Y]

def predict(data):
    #df1 = data[data.iloc[:, 9:19].sum(axis=1) == 0]
    data['predict'] = np.zeros(len(data))
    #df2 = data[data.iloc[:, 9:19].sum(axis=1) != 0]
    return data