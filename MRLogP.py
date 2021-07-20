import sys 
import os
import sklearn.utils
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense
from tensorflow.compat.v1.keras.layers import Dropout
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1.keras.callbacks import Callback
from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint
from tensorflow.compat.v1.keras.layers import PReLU
from tensorflow.compat.v1.keras.models import load_mode


class Model:
    def __init__(self):
        pass
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    def build_model (self, droprate, mid_h_layers, mid_h_nodes, learning_rate):
        K.clear_session()
        set_session(tf.Session(config=config))
        classifier = Sequential()

        # input layer and the first hidden layer
        classifier.add(Dense(units=mid_h_nodes, kernel_initializer='random_uniform', input_dim=316))
        classifier.add(PReLU())
        classifier.add(Dropout(rate=droprate))

        #hidden layers
        for r in range(mid_h_layers-1):  # add number of hidden layers
            classifier.add(Dense(units=mid_h_nodes))
            classifier.add(PReLU())
            classifier.add(Dropout(rate=droprate))

        classifier.add(Dense(units=1))
        classifier.add(PReLU())

        # output layers
        classifier.add(Dense(units=1, kernel_initializer='normal', activation='linear'))

        # compilation
        opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        classifier.compile(optimizer=opt, loss=[root_mean_squared_error], metrics=[root_mean_squared_error])
        return classifier

    def training(self):
        pass
    def cv(self):
        pass
    def transfer_learning(self):
        pass
    def predict_logP(self):
        pass
def main():


if __name__ == "__main__:
    main()

