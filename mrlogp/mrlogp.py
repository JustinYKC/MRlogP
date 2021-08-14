import os
from itertools import product as iterproduct
from numpy.lib.arraypad import _set_reflect_both
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
from ..smi_to_logP_descriptors import MRLogPDescriptor_Generator

class MRlogP():
    """
    MRlogP class, used to predict molecular logPs

    Consists of methods for generting features for training and testing from datasets 
    
    as well as for training and transfer learining models. 
    
    """   
    scaler=None
    y_class=None
    hyperparameter_options = {'droprate':[0.2], 
                              'hidden_layers':[1],
                              'hidden_nodes':[1264], 
                              'learning_rate':[0.0001],
                              'batch_size':[32,64], 
                              'epochs':[2],
                              }
    '''
    tl_parameter_options = {'epochs_for_output_layer':[1,2,3,4,5],
                            'epoch_for_tweaking':[1,2,3,4,5],
                            'learnrate_on_tweaking':[1.31E-5],
                            'unfrozen_layers':[2,1],
                            'batch_size':[64,128],
                            }
    '''
    tl_parameter_options = {'epochs_for_output_layer':[1,2],
                            'epoch_for_tweaking':[1,2],
                            'learnrate_on_tweaking':[1.31E-5],
                            'unfrozen_layers':[2,1],
                            'batch_size':[64],
                            }
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
        It returns four arrays which are the features for training, the features for validation, the labels
             for training, and the labels for validation.
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

    def create_testset(self, infile_testing:str, query_mode:bool=False):
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
        if query_mode == False:
            try: testset[['id', 'logP']] = testset.loc[:, "Name"].str.split(pat=';', expand=True)
            except: testset['logP'] = testset.loc[:, "Name"]
        else: testset['id'] = testset.loc[:, "Name"]

        x_ecfp4 = testset.loc[:, ["ecfp4-"+str(x) for x in range(128)]].astype('int64').astype('category').to_numpy()
        x_fp4 = testset.loc[:, ["fp4-"+str(x) for x in range(128)]].astype('int64').astype('category').to_numpy()
        x_usr = testset.loc[:, ["usrcat-"+str(x) for x in range(60)]].astype('float').to_numpy()
        if query_mode == False: y = testset.loc[:, 'logP'].astype('float').to_numpy()

        x_usr = self.scaler.transform(x_usr)
        x = np.hstack([x_ecfp4, x_fp4, x_usr])
        del testset

        if query_mode == False: return x, y
        else: return x
    
    def _create_cv_set(self, X_train:np.array, cv:int=10):
        """
        A private function used to generate train/validation indices to split data into n-folds (Default to 10 fold) by preserving the 
        proportion of each logP class (1 class: 1 log unit) in each training/validation set.  

        Parameters
        ----------
        X_train: (array, required)
            Molecular descriptors used for training. 
        
        cv: (int, optional)
            Number of folds. Default to 10.

        Returns
        -------
        It returns a dictionary containing each fold index and its corresponding train/validation indince arrays in a tuple.
        """
        kf = StratifiedKFold(n_splits=cv, shuffle=False, random_state=None)
        return {fold:split_index for fold, split_index in enumerate(kf.split(X_train, self.y_class))}

    def _handle_results(self, result_list:list, working_dir:Path):
        """
        A private function used to handle the results from hyperparameter scan, and fold cross validation; writting out the
        resulting a csv file and selecting the best models for further training process.

        Parameters
        ----------
        result_dict: (list, required)
            A list contains the given model infomation and their results from hyperparameter scan and fold cross validation. Each 
            model information and results is included in a dictionary.
        
        select_top: (int, optional)
            Number for models selected for further training. Default to 20.

        Returns
        -------
        It returns a list containing the model infomation required for further training course.
        """
        for index, each_model in enumerate(result_list):
            try:
                history_list = each_model["Results"]
            except: break
            for t, history in enumerate(history_list):
                rmse_train = history.history['root_mean_squared_error'][-1]
                col_name_rmse_train = f"RMSE_Train_{t+1}"
                result_list[index][col_name_rmse_train] = rmse_train
                try:
                    rmse_best_val = history.history['root_mean_squared_error'][np.argmin(history.history['val_loss'])]
                    col_name_rmse_bestvali = f"RMSE_BestVali_{t+1}"
                    result_list[index][col_name_rmse_bestvali] = rmse_best_val
                except: pass
        result_df = pd.DataFrame(result_list)
        if "Results" in result_df.columns: result_df.drop(columns='Results', inplace=True)
        result_df.to_csv(Path(working_dir)/Path("result.csv"), index=True, index_label="Index")
        #return result_df.sort_values("RMSE_BestVali").head(select_top).to_dict("records")

    def train(self, large_dataset:Path, val_split:float=0.1, hyperparameter_options:dict=None, cv:int=None, working_dir:Path=Path("./")):
        """
        Train the neural network models with given training set and the hyperparameters.  

        Parameters
        ----------
        large_dataset: (File path object, required)
            The path of the dataset for training

        val_split: (float, optional)    
            The fraction used to divide the dataset into a subset for validation in hyperparameter scan. Defaults to 0.1.

        hyperparameter_options: (dict, optional) 
            A dictionary contrains set hyperparameters used to create models for hyperparameter scan, cross validation, 
            and the final training. Defaults set to None which then diectly uses the hyperparameters of our best model: 
            (dropout=0.1, middle_hidden_layers=1, middle_hidden_nodes=1264, learning_rate=0.0001, batch_size=32, epochs=30)
        
        cv: (int, optional)
            Number of folds splitted for the use of cross validation. Default to None. 

        working_dir: (File path object, optional)
            The path where outputs the relevant results, including a csv file for results of hyperparameter scan, cross 
            validation and hdf5 files for saved models.

        Returns
        -------
        The trainig history from the process of the hyperparameter scan, cross validation, final trianing and transfer 
        learning. These resulting RMSEs are also summarised in csv files. 
        """
        if hyperparameter_options is None:
            hyperparameter_options = {
                'droprate':0.2,
                'hidden_layers':1,
                'hidden_nodes':1264,
                'learning_rate':0.0001,
                'batch_size':32,
                'epochs':30,
                }

        if cv:
            X_train_t, _, y_train_t, _ = self.create_training_set(infile_training=large_dataset, val_split=val_split)
            cv_index_dict = self._create_cv_set(X_train=X_train_t, cv=cv) 
            times = cv
        else: times = 1

        history_list=[]
        for i in range(times):
            if cv:
                X_train = X_train_t[cv_index_dict[i][0]]
                y_train = y_train_t[cv_index_dict[i][0]]
                X_val = X_train_t[cv_index_dict[i][1]]
                y_val = y_train_t[cv_index_dict[i][1]]
            else:
                X_train, X_val, y_train, y_val = self.create_training_set(infile_training=large_dataset, val_split=val_split)
            model = Model(droprate=hyperparameter_options['droprate'], mid_h_layers=hyperparameter_options['hidden_layers'], mid_h_nodes=hyperparameter_options['hidden_nodes'], learning_rate=hyperparameter_options['learning_rate'], batch_size=hyperparameter_options['batch_size'], working_dir=Path(working_dir))
            history = model.train(X_train, y_train, X_val, y_val, epochs=hyperparameter_options['epochs'])
            history_list.append(history)
            del model, X_train, X_val, y_train, y_val
        return history_list
            
    # Hyperparameter scan:
    def hyerparameter_scan(self, Larget_dataset:Path, hyperparameter_options:dict=None, val_split:float=0.1, working_dir:Path=Path("./hyperparameter_scan")):
        """
        Perform hyperparameter scan with given a set of hyperparameter combinations using gird search.   

        Parameters
        ----------
        large_dataset: (File path object, required) 
            The path of the dataset for performing hyperparameter scan
        
        hyperparameter_options: (dict, optional)
            A dictionary contrains set hyperparameters used to perform hyperparameter scan. Defaults set to None which then
            diectly uses the hyperparameters of our best model: 
            (dropout=0.1, middle_hidden_layers=1, middle_hidden_nodes=1264, learning_rate=0.0001, batch_size=32, epochs=30)

        val_split: (float, optional)    
            The fraction used to divide the dataset into a subset for validation in hyperparameter scan. Defaults to 0.1.

        working_dir: (File path object, optional)
            The path where outputs the results of hyperarameter scan, including a csv file for results of hyperparameter scan, 
            and hdf5 files for the models trained with each scaned hyperparameter set. Defaults set to the directory of 
            "./hyperparameter_scan".
        """
        if hyperparameter_options is None:
            hyperparameter_options = self.hyperparameter_options
        hyperparameter_options = [dict(zip(hyperparameter_options.keys(), ele)) for ele in iterproduct(*hyperparameter_options.values())]
        print (f"\nScanning {len(hyperparameter_options)} hyperparameter combinations...")
        for index, para_dict in enumerate(hyperparameter_options):
            p = Path(working_dir/f"{index+1}")
            p.mkdir(parents=True, exist_ok=True)
            hist_results_list = self.train(large_dataset=Larget_dataset, hyperparameter_options=para_dict, val_split=val_split, working_dir=Path(working_dir/f"{index+1}"))
            hyperparameter_options[index]["Model"] = index+1
            hyperparameter_options[index]["Results"] = hist_results_list
        self._handle_results(hyperparameter_options, working_dir)
        
    # N-fold CV:
    def cv(self, larget_dataset:Path, hyperparameter_options:dict=None, cv:int=10, working_dir:Path=Path("./cv")):
        """
        Perform cross validation with given a set of hyperparameter combinations.   

        Parameters
        ----------
        large_dataset: (File path object, required) 
            The path of the dataset for performing hyperparameter scan
        
        hyperparameter_options: (dict, optional)
            A dictionary contrains set hyperparameters used to perform cross validation. Defaults set to None which then
            diectly uses the hyperparameters of our best model: 
            (dropout=0.1, middle_hidden_layers=1, middle_hidden_nodes=1264, learning_rate=0.0001, batch_size=32, epochs=30)

        cv: (float, optional)    
            Number of folds splitted for the use of cross validation. Default to 10 folds.

        working_dir: (File path object, optional)
            The path where outputs the results of cross validation, including a csv file for results of cross validation,  
            hdf5 files for the models of each fold trained with each the given hyperparameter combination. Defaults set 
            to the directory of  "./cv".
        """
        if hyperparameter_options is None: hyperparameter_options = self.hyperparameter_options
        hyperparameter_options = [dict(zip(hyperparameter_options.keys(), ele)) for ele in iterproduct(*hyperparameter_options.values())]
        print (f"\nPerforming {cv}-fold CV...")
        for index, para_dict in enumerate(hyperparameter_options):
            p = Path(working_dir/f"{index+1}")
            p.mkdir(parents=True, exist_ok=True)
            hist_results_list = self.train(large_dataset=larget_dataset, hyperparameter_options=para_dict, val_split=0.0, cv=cv, working_dir=Path(working_dir/f"{index+1}"))
            hyperparameter_options[index]["Model"] = index+1
            hyperparameter_options[index]["Results"] = hist_results_list
        self._handle_results(hyperparameter_options, working_dir)
    
    #Final train against full large dataset and then test three druglike test sets 
    def final_train(self, larget_dataset:Path, small_precise_dataset:Path, reaxys_dataset:Path, physprop_dataset:Path, model_path:Path, hyperparameter_options:dict=None, working_dir:Path=Path("./final_training")):
        """
        Train the neural network models with the given hyperparameters against the full training set (None subset splitted
        for validation). The resulting model is then tested against 3 druglike data sets.

        Parameters
        ----------
        Large_dataset: (File path object, required)
            The path of the dataset for training

        small_precise_dataset: (File path object, required)
            The path of the dataset containing high quality measured logPs (i.e. Martel_DL). This dataset is used for testing 
            at the end of the training and for transfer learning. 

        reaxys_dataset: (File path object, required)
            The path of the Reaxys_DL test set used to evaluate the model at the end of the training and transfer learning.

        physprop_dataset: (File path object, required)
            The pathe of the Physprop_DL test set used to evaluate the model at the end of the training and transfer learning.
        
        hyperparameter_options: (dict, optional)
            A dictionary contrains set hyperparameters used to perform the final training. Defaults set to None which then
            diectly uses the hyperparameters of our best model: 
            (dropout=0.1, middle_hidden_layers=1, middle_hidden_nodes=1264, learning_rate=0.0001, batch_size=32, epochs=30)
        
        working_dir: (File path object, optional)
            The path where outputs the results of the final training, including a csv file for results of the final training, 
            and the test RMSEs of the resulting model against the 3 druglike datasets. The hdf5 file for the final model 
            trained with the given hyperparameter combination is also placed here. Defaults set to the directory of 
            "./final_training".
        """
        if hyperparameter_options is None: hyperparameter_options = self.hyperparameter_options
        hyperparameter_options = [dict(zip(hyperparameter_options.keys(), ele)) for ele in iterproduct(*hyperparameter_options.values())]
        print (f"\nTraining against full training set and then testing against three druglike sets...")
        for index, para_dict in enumerate(hyperparameter_options):
            p = Path(working_dir/f"{index+1}")
            p.mkdir(parents=True, exist_ok=True)
            hist_results_list = self.train(large_dataset=larget_dataset, hyperparameter_options=para_dict, val_split=0.0, working_dir=Path(working_dir/f"{index+1}"))
            hyperparameter_options[index]["Model"] = index+1
            hyperparameter_options[index]["Results"] = hist_results_list

        # Load final model for testing against three druglike test sets
        predictor = Model.load_predictor(model_path) 
        X_martel, y_martel = self.create_testset(small_precise_dataset)
        X_reaxys, y_reaxys = self.create_testset(reaxys_dataset)
        X_physprop, y_physprop = self.create_testset(physprop_dataset)
        rmse_martel = rmse(y_martel, predictor.predict(X_martel)).item(0)
        rmse_reaxys = rmse(y_reaxys, predictor.predict(X_reaxys)).item(0)
        rmse_physprop = rmse(y_physprop, predictor.predict(X_physprop)).item(0)
        hyperparameter_options[index]["RMSE_Martel_DL"] = rmse_martel
        hyperparameter_options[index]["RMSE_Reaxys_DL"] = rmse_reaxys
        hyperparameter_options[index]["RMSE_Physprop_DL"] = rmse_physprop
        self._handle_results(hyperparameter_options, working_dir)
        print (f"RMSE for Martel_DL: {rmse_martel:.3f}")
        print (f"RMSE for Reaxys_DL: {rmse_reaxys:.3f}")
        print (f"RMSE for Physprop_DL: {rmse_physprop:.3f}")

    #Transfer learning
    def transfer_learning(self, larget_dataset:Path, small_precise_dataset:Path, reaxys_dataset:Path, physprop_dataset:Path, pre_trained_model:Path, tl_parameter_options:dict=None, working_dir:Path=Path("./transfer_learning")):
        """
        Carry out transfer learning on a pre-trained model with given a set of transfer learning hyperparameters.    

        Parameters
        ----------
        Large_dataset: (File path object, required)
            The path of the dataset for training

        small_precise_dataset: (File path object, required)
            The path of the dataset containing high quality measured logPs (i.e. Martel_DL). This dataset is used to futher
            tweak the pre-trained model for higher prediction performance. 

        reaxys_dataset: (File path object, required)
            The path of the Reaxys_DL test set used to evaluate the model at the end of the transfer learning.

        physprop_dataset: (File path object, required)
            The path of the Physprop_DL test set used to evaluate the model at the end of the transfer learning.

        pre_trainied_model: (File path object, required)
            The path of the pre-trained model needed for transfer learing.
        
        tl_parameter_options: (dict, optional)
            A dictionary contrains set hyperparameters used to perform transfer learning. Defaults set to None which then
            diectly provide the hyperparameters used in our case for tuning the pre-trained model to get MRlogP:
            (epochs_for_output_layer':[1,2,3,4,5], 'epoch_for_tweaking':[1,2,3,4,5], 'learnrate_on_tweaking':[1.31E-5],
             'unfrozen_layers':[2,1],'batch_size':[64,128]).

        working_dir: (File path object, optional)
            The path where outputs the results of transfer learning, including a csv file for results of transfer training, 
            and the test RMSEs of the resulting model against the 3 druglike datasets.The hdf5 file for the model 
            transfer trained with the each of given hyperparameter combinations. Defaults set to the directory of 
            "./transfer_learning".
        """
        if tl_parameter_options is None: tl_parameter_options = self.tl_parameter_options
        tl_parameter_options = [dict(zip(tl_parameter_options.keys(), ele)) for ele in iterproduct(*tl_parameter_options.values())]
        print ("\nPerforming transfer learning - Loading the pre-trained model")
        #model_1 = Model.load_predictor(Path(pre_trained_model))
        _, _, _, _ = self.create_training_set(larget_dataset)
        X_martel, y_martel = self.create_testset(small_precise_dataset)
        X_reaxys, y_reaxys = self.create_testset(reaxys_dataset)
        X_physprop, y_physprop = self.create_testset(physprop_dataset)        

        print ("\nPerforming transfer learning - Starting fine-tune")
        for index, para_dict in enumerate(tl_parameter_options):
            p = Path(working_dir/f"{index+1}")
            p.mkdir(parents=True, exist_ok=True)
            model = Model(working_dir = Path(working_dir/f"{index+1}"))
            model.transfer_learning(X_martel, y_martel, pre_trained_model, para_dict["epochs_for_output_layer"], para_dict["epoch_for_tweaking"], para_dict["learnrate_on_tweaking"], para_dict["unfrozen_layers"], para_dict["batch_size"])

            classifier_tl = Model.load_predictor(Path(working_dir/f"{index+1}"/"model-tl.hdf5"))
            rmse_reaxys_tl = rmse(y_reaxys, classifier_tl.predict(X_reaxys)).item(0)
            rmse_physprop_tl = rmse(y_physprop, classifier_tl.predict(X_physprop)).item(0)
            tl_parameter_options[index]["Model"] = index+1
            tl_parameter_options[index]["TL_RMSE_Reaxys_DL"] = rmse_reaxys_tl
            tl_parameter_options[index]["TL_RMSE_Physprop_DL"] = rmse_physprop_tl
        self._handle_results(tl_parameter_options, working_dir)

    def predict_logp(self, larget_dataset:Path, query_csv_file:Path, model_path:Path):
        _, _, _, _ = self.create_training_set(larget_dataset)
        X_query = self.create_testset(query_csv_file, True)
        predictor = Model.load_predictor(model_path)
        print (predictor.predict(X_query))