from tensorflow.compat.v1.keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def rmse(y_true, y_pred):
    res=0
    for i in range(len(y_true)):
        res=res+((y_true[i]-y_pred[i])**2)
    res=res/len(y_true)
    res=res**0.5
    return res
