import numpy as np
import scipy.stats


def mean_confidence_interval(data, name='', confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.norm.ppf((1 + confidence) / 2.)
    
    to_return = f'{name} \n {round(m, 4)} $\pm$ {round(se, 4)} ({round((m-h), 4)} - {round((m+h), 4)})'
    
    return to_return