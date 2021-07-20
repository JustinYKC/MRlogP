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
    def __init__(self, droprate=0.1, mid_h_layers=1, mid_h_nodes=1264, learning_rate=0.001):
        self.droprate = droprate
        self.h_layers = mid_h_layers
        self.h_nodes = mid_h_nodes
        self.lr = learning_rate
        self.classiifer = None
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    def build_model (self, droprate, mid_h_layers, mid_h_nodes, learning_rate):
        K.clear_session()
        set_session(tf.Session(config=config))
        self.classiifer = Sequential()
        # input layer and the first hidden layer
        self.classiifer.add(Dense(units=self.h_nodes, kernel_initializer='random_uniform', input_dim=316))
        self.classiifer.add(PReLU())
        self.classiifer.add(Dropout(rate=self.droprate))
        #hidden layers
        for r in range(self.h_layers-1):  # add number of hidden layers
            self.classiifer.add(Dense(units=self.h_nodes))
            self.classiifer.add(PReLU())
            self.classiifer.add(Dropout(rate=self.droprate))
        self.classiifer.add(Dense(units=1))
        self.classiifer.add(PReLU())
        # output layers
        self.classiifer.add(Dense(units=1, kernel_initializer='normal', activation='linear'))
        # compilation
        opt = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.classiifer.compile(optimizer=opt, loss=[root_mean_squared_error], metrics=[root_mean_squared_error])
        self.classiifer.summary()
        return self.classiifer

    def training(self, reapeats, X_train, y_train, X_test, y_test, chunksize):
        for n in reapeats:
            checkpointer = [ModelCheckpoint(os.path.join(current_dir, "model-"+str(no_model)+str(fold)+"_bestValidation_beforeTL.hdf5"), monitor='val_loss', verbose=0, save_best_only=True, mode='min')]
            history = classifier.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=chunksize, callbacks=[checkpointer])
            classifier.save(os.path.join(current_dir, "model-"+str(no_model)+str(fold)+"_endTraining_beforeTL.hdf5"))
        
    def cv(self):
        pass
    def transfer_learning(self):
        pass
    def predict_logP(self):
        pass
def main():


if __name__ == "__main__:
    main()

