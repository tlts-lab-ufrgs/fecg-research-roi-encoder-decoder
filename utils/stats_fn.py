import numpy as np
import scipy.stats


def mean_confidence_interval(data, name='', confidence=0.95, percentage=False):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.norm.ppf((1 + confidence) / 2.)

    if percentage:
        m *= 100
        se *= 100
    
    to_return = f'{name} \n {round(m, 4)} $\pm$ {round(se, 4)} ({round((m-h), 4)} - {round((m+h), 4)})'
    
    return to_return

def mae_function(y_true, y_pred):
    
    mae_value = np.mean(
        np.abs((y_true - y_pred)) # , 2)
    )
    
    return mae_value

def mse_function(y_true, y_pred):
    
    mse_value = np.mean(
        np.power((y_true - y_pred) , 2)
    )
    
    return mse_value