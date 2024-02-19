import numpy as np

def mse_function(y_true, y_pred):
    
    mse_value = np.mean(
        np.power(y_true - y_pred, 2)
    )
    
    return mse_value