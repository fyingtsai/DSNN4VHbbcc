from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

import argparse
import os
import DSNNr as DSNNr
import math
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from cycler import cycler
rc("text", usetex=False)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
def train(args):
    if not os.path.isdir(args.global_name):
        os.makedirs(args.global_name)
    os.chdir(args.global_name)

    with open("config.txt", "w") as f:
        f.write(str(args))
        f.close()
    isTraining = args.isTraining
    args.MCa_path = args.MCa
    args.MCb_path = args.MCb
    if not isTraining:
       (MCa, MCb, MCa_spec, MCb_spec, MCa_weights, MCb_weights, maxObjCount)  = DSNNr.get_data(args)
    if isTraining:
       inputOriaFile = "SherpaMCa.hadd.npz"
       inputOribFile = "MadgaphMCa.hadd.npz"
       MCa = np.load(inputOriaFile)["MCa"]
       MCb = np.load(inputOribFile)["MCb"]
       MCa_weights = np.load(inputOriaFile)["MCa_weights"]
       MCb_weights = np.load(inputOribFile)["MCb_weights"]
       MCa_spec = np.load(inputOriaFile)["MCa_spec"]
       MCb_spec = np.load(inputOribFile)["MCb_spec"]
    
    
       (
           X_train,
           X_test,
           Y_train,
           Y_test,
           train_weights,
           test_weights,
           S_train,
           S_test,
           class_weights,
       ) = DSNNr.handle_data(args, MCa, MCb, MCa_weights, MCb_weights, MCa_spec, MCb_spec)
    
    #model = DSNNr.basic_model(args, n_features)
       n_features = len(args.features.split(","))
       model = DSNNr.DS_model(n_features) 
       checkpoint = ModelCheckpoint('./saved_models/'+ "/model-{epoch:03d}.ckpt",
                                    monitor='val_loss',
                                    verbose=2,
                                    save_freq='epoch',
                                    save_best_only=False,
                                    save_weights_only=False,
                                    mode='min')
       csvLogger = CSVLogger("trainingCSV.csv", separator=",", append=False)
       earlyStopping = EarlyStopping(monitor='val_loss', 
                                     min_delta=0, 
                                     patience=20, 
                                     verbose=1, 
                                     restore_best_weights=True)
    
       callbacks = [checkpoint, csvLogger, earlyStopping]
       # -----------
       # Train model
       # -----------
       start_train = time.time()      
       history = model.fit(X_train, Y_train,
                           epochs = 200,
                           batch_size = 50000,
                           validation_data = (X_test, Y_test, test_weights),
                           class_weight=class_weights,
                           sample_weight=train_weights,
                           verbose = 1, 
                           callbacks = callbacks) # Train the model with the new callback
       end_train = time.time()
       #print('Y_test:',Y_test)
       print("Time consumed in training: ",end_train - start_train)
    

if __name__ == "__main__":
    args = DSNNr.handle_args()
    train(args)
