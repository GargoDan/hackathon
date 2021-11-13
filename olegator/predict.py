import pandas as pd
from scipy.stats import randint, norm

def make_bootstrap(X):
    Y = randint.rvs(low = 0, high = len(X),  size = len(X))
    
    return [X[i] for i in Y]

def predict(data):
    df1 = data[data.iloc[:, 9:19].sum(axis=1) == 107]
    df1['predict'] = np.zeros(len(df1))
    df2 = data[data.iloc[:, 9:19].sum(axis=1) != 0]
    return df1