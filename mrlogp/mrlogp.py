 import os
from itertools import product as iterproduct
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
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
from pathlib import Path
from .metrics import *
from .mrlogp_model import Model

class MRlogP():
    """
    MRlogP class, used to predict molecular logPs

    Consists of methods for generting features for training and testing from datasets 
    
    as well as for training and transfer learining models. 
    
    """   
    scaler=None
    y_class=None

    def __init__(self) -> None:
        pass

    def create_training_set(self, infile_training:str, val_split:float=0.1):
        """
        Create MRlogP training and validation set

        Parameters
        ----------
        infile_training: (str, required)
            The path of the dataset for training and validation 

        val_split: (float, optional) 
            The fraction to divide the dataset into the subset used for validation. Defaults to 0.1.

        Returns
        -------
        It returns five arrays. The first four arrays repesent features for training, features for validation, 
            labels for training, and labels for validation. The last one shows the distribution of logP classes 
            (1 class: 1 log unit) in the dataset. 
        """       
        if isinstance(infile_training, str):
            infile_training = Path(infile_training)
        if not infile_training.exists():
            raise FileNotFoundError(f"Training input file ({infile_training} not found)")
        input_csv = pd.read_csv(infile_training)
        dataset = shuffle(input_csv)
        dataset[['id', 'logP']] = dataset.loc[:, "Name"].str.split(pat=',', expand=True)
        dataset[['logP', "-"]] = dataset.loc[:, "logP"].str.split(pat='_', expand=True)
        x_ecfp4 = dataset.loc[:, ["ecfp4-"+str(x) for x in range(128)]].astype('int64').astype('category').to_numpy()
        x_fp4 = dataset.loc[:, ["fp4-"+str(x) for x in range(128)]].astype('int64').astype('category').to_numpy()
        x_usr = dataset.loc[:, ["usrcat-"+str(x) for x in range(60)]].astype('float').to_numpy()
        y = dataset.loc[:, 'logP'].astype('float').to_numpy()
        cpd_name = dataset.loc[:, 'Name'].astype('str').to_numpy()
    
        self.scaler = StandardScaler().fit(x_usr)
        x_usr = self.scaler.transform(x_usr)
        x = np.hstack([x_ecfp4, x_fp4, x_usr])
        del dataset

        #Create validation set by sampling binned logP proportional to the logP bins in the training set
        self.y_class = y
        for index, bin in enumerate(range (-5, 10, 1)):
            #print (f"{index}, {bin} <= bi < {bin+1}, {np.where((y_train >= bin) & (y_train < bin+1))}, {y_train[np.where((y_train >= bin) & (y_train < bin+1))]}, {len(y_train[np.where((y_train >= bin) & (y_train < bin+1))])}")
            X_train_sub = x[np.where((y >= bin) & (y < bin+1))]
            y_train_sub = y[np.where((y >= bin) & (y < bin+1))]
            cpd_id_sub = cpd_name[np.where((y >= bin) & (y < bin+1))]
            self.y_class = np.where((y >= bin) & (y < bin+1), index, self.y_class)
            X_train_1, X_test_1, y_train_1, y_test_1, _, _ = train_test_split(X_train_sub, y_train_sub, cpd_id_sub, test_size=val_split, random_state=0)
            if index == 0:
                X_train_p = X_train_1
                X_test_p = X_test_1
                y_train_p = y_train_1 
                y_test_p = y_test_1
            else:
                X_train_p = np.concatenate((X_train_p, X_train_1), axis=0)
                X_test_p = np.concatenate((X_test_p, X_test_1), axis=0)
                y_train_p = np.concatenate((y_train_p, y_train_1)) 
                y_test_p = np.concatenate((y_test_p, y_test_1))

        return X_train_p, X_test_p, y_train_p, y_test_p

    def create_testset(self, infile_testing:str):
        """
        Create a MRlogP test set

        Parameters
        ----------
        infile_training: (str, required)
            The path of the dataset for testing 

        Returns
        -------
        It returns two arrays repesenting features and labels for testing.
        """
        testset = pd.read_csv(infile_testing)
        try:
            testset[['id', 'logP']] = testset.loc[:, "Name"].str.split(pat=';', expand=True)
        except:
            testset['logP'] = testset.loc[:, "Name"]

        x_ecfp4 = testset.loc[:, ["ecfp4-"+str(x) for x in range(128)]].astype('int64').astype('category').to_numpy()
        x_fp4 = testset.loc[:, ["fp4-"+str(x) for x in range(128)]].astype('int64').astype('category').to_numpy()
        x_usr = testset.loc[:, ["usrcat-"+str(x) for x in range(60)]].astype('float').to_numpy()
        y = testset.loc[:, 'logP'].astype('float').to_numpy()

        x_usr = self.scaler.transform(x_usr)
        x = np.hstack([x_ecfp4, x_fp4, x_usr])
        del testset

        return x, y
    
    def _create_cv_set(self, X_train, y_train, cv=10):
        fold=0
        kf = StratifiedKFold(n_splits=cv, shuffle=False)
        for train_index, test_index in kf.split(X_train, self.y_class):
            X_train_fold = X_train[train_index]
            y_train_fold = y_train[train_index]
            X_val_fold = X_train[test_index]
            y_val_fold = y_train[test_index]

            checkpointer = [ModelCheckpoint(os.path.join(self.work_dir, f"model_{fold}_bestValidation_fromCV.hdf5"), monitor='val_loss', verbose=0, save_best_only=True, mode='min')]
            history = self.classifier.fit(X_train_fold, y_train_fold, epochs=epochs, validation_data=(X_val_fold, y_val_fold), batch_size=self.chunksize, callbacks=[checkpointer])
    def _handle_results(self, col): 
        pass

    def train(self, large_dataset:Path, small_precise_dataset:Path, reaxys_dataset:Path, physprop_dataset:Path, val_split:float, hyperparameter_options:dict=None, cv:int=10):
        """
        Train the neural network models with the given hyperparameters using training set  

        Parameters
        ----------
        Large_dataset: (File path object, required)
            The path of the dataset for training
        
        small_precise_dataset: (File path object, required)
            The path of the dataset containing high quality measured logPs (i.e. Martel_DL). This dataset is used for testing at 
            the end of the training and for transfer learning. 

        reaxys_dataset: (File path object, required)
            The path of the Reaxys_DL test set used to evaluate the model at the end of the training and transfer learning.

        physprop_dataset: (File path object, required)
            The pathe of the Physprop_DL test set used to evaluate the model at the end of the training and transfer learning.

        val_split: (float, optional)    
            The fraction to divide the dataset into the subset used for validation. Defaults to 0.1.

        hyperparameter_options: (dict, optional) 
            A dictionary contrains set hyperparameters used to create models. Defaults set to the hyperparameters of our best 
            model: (dropout=0.1, middle_hidden_layers=1, middle_hidden_nodes=1264, learning_rate=0.0001, batch_size=32, epochs=30)
        
        cv: (int, optionaal)


        Returns
        -------
        The RMSEs tested against the given test sets from hyperparameter scan, cross validation, final trianing and transfer learning. 
        These resulting RMSEs are also summarised in csv files. 
        """
        X_train, X_val, y_train, y_val = self.create_training_set(infile_training=large_dataset, val_split=val_split)
        
        if hyperparameter_options is None:
            hyperparameter_options={
                'droprate':0.2,
                'hidden_layers':1,
                'hidden_node':1264,
                'learning_rate':0.0001,
                'batch_size':32,
                'epochs':30,
                }
        #Hyperparameter scan:
        result_list 
        model = Model(droprate=hyperparameter_options['droprate'], mid_h_layers=hyperparameter_options['hidden_layers'], mid_h_nodes=hyperparameter_options['hidden_nodes'], learning_rate=hyperparameter_options['learning_rate'], batch_size=hyperparameter_options['batch_size'], working_dir=Path("data"))
        #model.cv(X_train=X_train, y_train=y_train, cv=10, epochs=para_list[5])
        model.train(X_train, y_train, X_val, y_val, epochs=hyperparameter_options['epochs'])
        
        #n-fold cv:
        X_train, X_val, y_train, y_val, y_class = self.create_training_set(infile_training=large_dataset, val_split=0.0)
        self._create_cv_set(X_train=X_train, y_train=y_train, cv=cv)
        
        #Final training against the whole large dataset
        self.train(X_train=X_train, y_train=y_train, epochs=hyperparameter_options['epochs'])

        # Load final model for testing against three druglike test sets
        classifier = Model.load_predictor() #XXX
        X_martel, y_martel = self.create_testset(small_precise_dataset)
        X_reaxys, y_reaxys = self.create_testset(reaxys_dataset)
        X_physprop, y_physprop = self.create_testset(physprop_dataset)
        rmse_martel = rmse(y_martel, classifier.predict(X_martel))
        rmse_reaxys = rmse(y_reaxys, classifier.predict(X_reaxys))
        rmse_physprop = rmse(y_physprop, classifier.predict(X_physprop))
        print (f"RMSE for Martel_DL: {rmse_martel}")
        print (f"RMSE for Reaxys_DL: {rmse_reaxys}")
        print (f"RMSE for Physprop_DL: {rmse_physprop}")


        model.transfer_learning(X_train=X_train, y_train=y_train, cv=cv)
    '''   
    def transfer_learning(self, pre_trained_model=Path, small_precise_dataset:Path, reaxys_dataset:Path, physprop_dataset:Path, tl_parameter_options:dict=None):
        #Transfer learning
        print ("\nPerforming transfer learning - Loading the pre-trained model")
        model = Model.load_from_file(Path("model.hdf5"))
        print ("\nPerforming transfer learning - Starting fine-tune")
    '''
if __name__ == "__main__":
    with tf.device("cpu:0"):
        main(infile_training="/home/justin/cnnLogP/logP-2021/dnn_training_tl_training_against_dlSet_2021/proportionalTestset_repeats/6/ds-descriptors-eMols201905-DL-500k-flat_FP4.csv", 
             infile_testing="/home/justin/cnnLogP/logP-2021/ds-descriptors-physprop_3d_allowed_atoms_FP4_qed_dl.csv",
             working_dir="/home/justin/cnnLogP/logP-2021/dnn_training_tl_training_against_dlSet_2021/proportionalTestset_repeats/model_deployment/dnnLogP-2021-model_20210716_model_github",
             model_file="/home/justin/cnnLogP/logP-2021/dnn_training_tl_training_against_dlSet_2021/transfer_learning_correct_2021/Architecture_1/model_294-2/4/model-tl-11.hdf5", 
             train=True, transfer_learn=False, predict_logP=True, val_split=0.1, droprate_list=[0.1], hidden_layer_list=[1], hidden_node_list=[1264], learnrate_list=[0.001], 
             chunksize_list=[32], epoch_list=[1,2], repeats=3, cv=None)