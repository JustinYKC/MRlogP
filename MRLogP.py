import sys 
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
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
from tensorflow.compat.v1.keras.models import load_model

class Dataset_maker():
    def __init__(self):
        self.X = None
        self.y = None 
        self.X_val = None
        self.y_val = None
        self.cpd_name = None
        self.cpd_name_val = None

    def create_training_set(self, input_file, val_split=0.1):
     with tf.device("cpu:0"):
        input_csv = pd.read_csv(input_file)
        dataset = shuffle(input_csv)
        dataset[['id', 'logP']] = dataset.loc[:, "Name"].str.split(pat=',', expand=True)
        dataset[['logP', "-"]] = dataset.loc[:, "logP"].str.split(pat='_', expand=True)
        x_ecfp4 = dataset.loc[:, ["ecfp4-"+str(x) for x in range(128)]].astype('int64').astype('category').to_numpy()
        x_fp4 = dataset.loc[:, ["fp4-"+str(x) for x in range(128)]].astype('int64').astype('category').to_numpy()
        x_usr = dataset.loc[:, ["usrcat-"+str(x) for x in range(60)]].astype('float').to_numpy()
        y = dataset.loc[:, 'logP'].astype('float').to_numpy()
        cpd_name = dataset.loc[:, 'Name'].astype('str').to_numpy()
 
        scaler = StandardScaler().fit(x_usr)
        x_usr = scaler.transform(x_usr)
        x = np.hstack([x_ecfp4, x_fp4, x_usr])
        del dataset

        #Create validation set by sampling binned logP proportional to the logP bins in the training set
        for index, bin in enumerate(range (-5, 10, 1)):
            #print (f"{index}, {bin} <= bi < {bin+1}, {np.where((y_train >= bin) & (y_train < bin+1))}, {y_train[np.where((y_train >= bin) & (y_train < bin+1))]}, {len(y_train[np.where((y_train >= bin) & (y_train < bin+1))])}")
            X_train_sub = x[np.where((y >= bin) & (y < bin+1))]
            y_train_sub = y[np.where((y >= bin) & (y < bin+1))]
            cpd_id_sub = cpd_name[np.where((y >= bin) & (y < bin+1))]
            X_train_1, X_test_1, y_train_1, y_test_1, cpd_name_train_1, cpd_name_test_1 = train_test_split(X_train_sub, y_train_sub, cpd_id_sub, test_size=val_split, random_state=0)
            if index == 0:
                X_train_p = X_train_1
                X_test_p = X_test_1
                y_train_p = y_train_1 
                y_test_p = y_test_1
                cpd_name_train_p = cpd_name_train_1
                cpd_name_test_p = cpd_name_test_1
            else:
                X_train_p = np.concatenate((X_train_p, X_train_1), axis=0)
                X_test_p = np.concatenate((X_test_p, X_test_1), axis=0)
                y_train_p = np.concatenate((y_train_p, y_train_1)) 
                y_test_p = np.concatenate((y_test_p, y_test_1))
                cpd_name_train_p = np.concatenate((cpd_name_train_p, cpd_name_train_1))
                cpd_name_test_p = np.concatenate((cpd_name_test_p, cpd_name_test_1))
            #print (f"{index}, {bin} <= bin < {bin+1}, {cpd_name_train_1}, {cpd_name_test_1}, {y_train_1}, {y_test_1}, {len(X_train_1)}, {len(X_test_1)}")

        self.X = X_train_p
        self.X_val = X_test_p
        self.y = y_train_p
        self.y_val = y_test_p
        self.cpd_name = cpd_name_train_p
        self.cpd_name_val = cpd_name_test_p
        return self.X, self.X_val, self.y, self.y_val, self.cpd_name, self.cpd_name_val, scaler

    def create_testset(self, inputfile, sc):
     with tf.device("cpu:0"):
        testset = pd.read_csv(inputfile)
        try:
            testset[['id', 'logP']] = testset.loc[:, "Name"].str.split(pat=';', expand=True)
        except:
            testset['logP'] = testset.loc[:, "Name"]

        x_ecfp4 = testset.loc[:, ["ecfp4-"+str(x) for x in range(128)]].astype('int64').astype('category').to_numpy()
        x_fp4 = testset.loc[:, ["fp4-"+str(x) for x in range(128)]].astype('int64').astype('category').to_numpy()
        x_usr = testset.loc[:, ["usrcat-"+str(x) for x in range(60)]].astype('float').to_numpy()
        y = testset.loc[:, 'logP'].astype('float').to_numpy()

        x_usr = sc.transform(x_usr)
        x = np.hstack([x_ecfp4, x_fp4, x_usr])
        del testset

        self.X = x
        self.y = y
        return self.X, self.y

class Model:
    def __init__(self, droprate=0.1, mid_h_layers=1, mid_h_nodes=1264, learning_rate=0.001, batch_size=32, working_dir="/home/justin/cnnLogP/logP-2021/dnn_training_tl_training_against_dlSet_2021/proportionalTestset_repeats/model_deployment/dnnLogP-2021-model_20210716_model_github"):
        self.droprate = droprate
        self.h_layers = mid_h_layers
        self.h_nodes = mid_h_nodes
        self.lr = learning_rate
        self.chunksize = batch_size
        self.classifier = None
        self.work_dir = f"{working_dir}"

    def build_model(self):
        K.clear_session()
        self.classifier = Sequential()
        # input layer and the first hidden layer
        self.classifier.add(Dense(units=self.h_nodes, kernel_initializer='random_uniform', input_dim=316))
        self.classifier.add(PReLU())
        self.classifier.add(Dropout(rate=self.droprate))
        #hidden layers
        for r in range(self.h_layers-1):  # add number of hidden layers
            self.classifier.add(Dense(units=self.h_nodes))
            self.classifier.add(PReLU())
            self.classifier.add(Dropout(rate=self.droprate))
        self.classifier.add(Dense(units=1))
        self.classifier.add(PReLU())
        # output layers
        self.classifier.add(Dense(units=1, kernel_initializer='normal', activation='linear'))
        # compilation
        opt = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.classifier.compile(optimizer=opt, loss=[root_mean_squared_error], metrics=[root_mean_squared_error])
        self.classifier.summary()
        return self.classifier

    def training(self, X_train, y_train, X_val, y_val, reapeats=2, epochs=1):
      with tf.device("cpu:0"): 
        for n in range(reapeats):
            checkpointer = [ModelCheckpoint(os.path.join(self.work_dir, "model-"+str(n)+"-repeat_bestValidation_fromTraining.hdf5"), monitor='val_loss', verbose=0, save_best_only=True, mode='min')]
            history = self.classifier.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=self.chunksize, callbacks=[checkpointer])
            #classifier.save(os.path.join(current_dir, "model-"+str(no_model)+str(fold)+"_endTraining_beforeTL.hdf5"))

    def cv(self):
        pass
    def transfer_learning(self):
        pass
    def predict_logP(self):
        pass
    def load_predictor(self, model_file):
     with tf.device("cpu:0"):
        self.classifier = load_model(model_file, custom_objects={'root_mean_squared_error': root_mean_squared_error})
        return self.classifier

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
def root_mean_squared_error_np(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))
def rmse(y_true, y_pred):
    res=0
    #print (y_true.shape, y_pred.shape)
    for i in range(len(y_true)):
        #print (y_true[i], y_pred[i])
        res=res+((y_true[i]-y_pred[i])**2)
    res=res/len(y_true)
    res=res**0.5
    return res
def main():
    pass

if __name__ == "__main__":
    with tf.device("cpu:0"):
        model = Model()
        #classifier = model.build_model()
        #print (classifier)
        data = Dataset_maker()
        X_train, X_val, y_train, y_val, _, _, sc = data.create_training_set("/home/justin/cnnLogP/logP-2021/dnn_training_tl_training_against_dlSet_2021/proportionalTestset_repeats/6/ds-descriptors-eMols201905-DL-500k-flat_FP4.csv")
        print (X_train.shape, X_val.shape, y_train.shape, y_val.shape)

        #model.training(X_train, y_train, X_val, y_val)


        #X_martel_dl, y_martel_dl = data.create_testset("/home/justin/cnnLogP/logP-2021/ds-descriptors-martel_FP4_qed_dl.csv", sc)
        del  X_train, X_val, y_train, y_val
        X_physprop_dl, y_physprop_dl = data.create_testset("/home/justin/cnnLogP/logP-2021/ds-descriptors-physprop_3d_allowed_atoms_FP4_qed_dl.csv", sc)
        #print (X_martel_dl.shape, y_martel_dl, y_martel_dl.shape)
        #print (X_physprop_dl.shape, y_physprop_dl.shape)

        classifier = model.load_predictor("/home/justin/cnnLogP/logP-2021/dnn_training_tl_training_against_dlSet_2021/transfer_learning_correct_2021/Architecture_1/model_294-2/4/model-tl-11.hdf5")
        rmse_result = rmse(y_physprop_dl, classifier.predict(X_physprop_dl))
        print (rmse_result)
        r=root_mean_squared_error_np(y_physprop_dl, classifier.predict(X_physprop_dl))
        print (K.eval(r))
        #main()
