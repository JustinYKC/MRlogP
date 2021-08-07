from numpy.core.fromnumeric import shape
from sklearn.utils import validation
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
import numpy as np
from .metrics import *
import time, os
from pathlib import Path

#XXX Check imports above are actually used
#XXX Add data types to function arguments

class Model(object):
    """
    Model class, used to build models.

    This class contains detailed settings for building model creation for training and transfer learining 
    as well as loading an existing model for further use.
    
    """  
    def __init__(self, working_dir, droprate=0.2, mid_h_layers=1, mid_h_nodes=1264, learning_rate=0.0001, batch_size=32):
        self.droprate = droprate
        self.h_layers = mid_h_layers
        self.h_nodes = mid_h_nodes
        self.lr = learning_rate
        self.chunksize = batch_size
        self.classifier = None
        self.work_dir = working_dir
        
        self._build_classifier()

    def _build_classifier(self):
        """
        A private function used to create a sequential neural network.
 
        """       
        K.clear_session()
        self.classifier = Sequential()
        # input layer and the first hidden layer
        self.classifier.add(Dense(units=self.h_nodes, kernel_initializer='random_uniform', input_dim=316))
        self.classifier.add(PReLU())
        self.classifier.add(Dropout(rate=self.droprate))
        #hidden layers
        for r in range(self.h_layers):  # add number of hidden layers
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
        #return self.classifier

    def train(self, X_train:np.array, y_train:np.array, X_val:np.array=None, y_val:np.array=None, epochs:int=30):
        """
        Train and validate models against the given training and validation set.

        Parameters
        ----------
        X_train: (array, required)
            Molecular descriptors used for training. 

        y_train: (array, required) 
            Labels used in training.

        X_val: (None or array, optional)
            Molecular descriptors used for validation.

        y_val: (None or array, optional)
            Labels used for validation.

        epochs: (int, optional)
            Numbers set for training epochs. Defaults to 30.
        """
        if X_val.shape[0] == 0 & y_val.shape[0] == 0:
            validation_data = None
        else: validation_data = (X_val, y_val)
        checkpointer = [ModelCheckpoint(os.path.join(self.work_dir, "model_bestValidation.hdf5"), monitor='val_loss', verbose=0, save_best_only=True, mode='min')]
        history = self.classifier.fit(X_train, y_train, epochs=epochs, validation_data=validation_data, batch_size=self.chunksize, callbacks=[checkpointer])
        self.classifier.save(os.path.join(self.work_dir, "model_endTraining.hdf5"))
        return history

    def transfer_learning(self, X_train, y_train, pre_train_model, epoch_1, epoch_2, lr_on_tweaking, unfr_layer, chunksize):
        #Create a new model by resuing all the layers but the output later from the base model  
        model_1 = Model.load_predictor(Path(pre_train_model))
        model_2_on_1 = Sequential(model_1.layers[:-1])

        #Freeze all the reused layers but the output later in the new model
        for _layer in model_2_on_1.layers:
            _layer.trainable = False
        #model_2_on_1(training = False)
        model_2_on_1.summary()

        #Add a new output layer to the new model
        model_2_on_1.add(Dense(units=1, kernel_initializer='normal', activation='linear', name='dense_3'))

        #Compile the new model
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model_2_on_1.compile(optimizer=opt, loss=[root_mean_squared_error], metrics=[root_mean_squared_error])
        model_2_on_1.summary()
        print ("\n-----------------------------------------------------------------------------------------------")
        print (f"Epochs for training new output layer:{epoch_1}, No. unfrozen layers:{unfr_layer}, Epochs for fine-tuning unfrozen layers:{epoch_2}")
        time.sleep(5) # Allow the user to read the terminal output

        #Optimise the new output layer
        model_2_on_1.fit(X_train, y_train, epochs=epoch_1)

        #Unfreeze reused layers
        for _layer in model_2_on_1.layers[-2:(-3*unfr_layer)-1:-1]: 
            _layer.trainable = True
        #model_2_on_1(training = True)

        #Compile the new model
        opt = Adam(lr=lr_on_tweaking, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model_2_on_1.compile(optimizer=opt, loss=[root_mean_squared_error], metrics=[root_mean_squared_error])
        model_2_on_1.summary()

        #Tweak the reused layers and save the model
        model_2_on_1.fit(X_train, y_train, epochs=epoch_2, batch_size=chunksize)
        model_2_on_1.save(os.path.join(self.work_dir, "model-tl.hdf5"))
        
    @staticmethod
    def load_predictor(model_file:Path):
        # XXX Check if string, if so, cast to path, check file exists, if not raise error.
            #model=Model(model_file.parent)
            classifier = load_model(model_file, custom_objects={'root_mean_squared_error': root_mean_squared_error})
            return classifier