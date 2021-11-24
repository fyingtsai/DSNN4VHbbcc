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
    '''
    numpyMCa1 = np.load("/hpcgpfs01/scratch/ftsai/DSNNrBranch/Reco_WJet_MCade_5050_Reweight/SherpaMCa.hadd.npz")
    numpyMCb1 = np.load("/hpcgpfs01/scratch/ftsai/DSNNrBranch/Reco_WJet_MCade_5050_Reweight/MadgaphMCa.hadd.npz")
    numpyMCa2 = np.load("/hpcgpfs01/scratch/ftsai/DSNNrBranch/Reco_WJet_MCade_5050_Reweight/SherpaMCd.hadd.npz")
    numpyMCb2 = np.load("/hpcgpfs01/scratch/ftsai/DSNNrBranch/Reco_WJet_MCade_5050_Reweight/MadgraphMCd.hadd.npz")
    numpyMCa3 = np.load("/hpcgpfs01/scratch/ftsai/DSNNrBranch/Reco_WJet_MCade_5050_Reweight/SherpaMCe.hadd.npz")
    numpyMCb3 = np.load("/hpcgpfs01/scratch/ftsai/DSNNrBranch/Reco_WJet_MCade_5050_Reweight/MadgraphMCe.hadd.npz")

    MCa1 = numpyMCa1["MCa"]
    MCb1 = numpyMCb1["MCb"]
    MCa2 = numpyMCa2["MCa"]
    MCb2 = numpyMCb2["MCb"]
    MCa3 = numpyMCa3["MCa"]
    MCb3 = numpyMCb3["MCb"]

    Wa1 =  numpyMCa1["MCa_weights"]
    Wb1 =  numpyMCb1["MCb_weights"]
    Wa2 =  numpyMCa2["MCa_weights"]
    Wb2 =  numpyMCb2["MCb_weights"]
    Wa3 =  numpyMCa3["MCa_weights"]
    Wb3 =  numpyMCb3["MCb_weights"]

    Sa1 = numpyMCa1["MCa_spec"]
    Sb1 = numpyMCb1["MCb_spec"]
    Sa2 = numpyMCa2["MCa_spec"]
    Sb2 = numpyMCb2["MCb_spec"]
    Sa3 = numpyMCa3["MCa_spec"]
    Sb3 = numpyMCb3["MCb_spec"]

    MCa = np.array([*MCa1, *MCa2, *MCa3])
    MCb = np.array([*MCb1, *MCb2, *MCb3])
    MCa_weights = np.array([*Wa1, *Wa2, *Wa3])
    MCb_weights = np.array([*Wb1, *Wb2, *Wb3])
    MCa_spectors = np.array([*Sa1, *Sa2, *Sa3])
    MCb_spectors = np.array([*Sb1, *Sb2, *Sb3])
    np.savez("CombinedMCade5050_reweight", MCa=MCa, MCb=MCb, MCa_weights=MCa_weights, MCb_weights=MCb_weights, MCa_spectors=MCa_spectors, MCb_spectors=MCb_spectors)
    '''
    '''
    SplitMCa = np.load("/hpcgpfs01/scratch/ftsai/DSNNrBranch/Reco_WJet_MCade_5050_Reweight/DataMCaSplit5050_NoLumi_split.npz")
    SplitMCd = np.load("/hpcgpfs01/scratch/ftsai/DSNNrBranch/Reco_WJet_MCade_5050_Reweight/DataMCdSplit5050_NoLumi_split.npz")
    SplitMCe = np.load("/hpcgpfs01/scratch/ftsai/DSNNrBranch/Reco_WJet_MCade_5050_Reweight/DataMCeSplit5050_NoLumi_split.npz")
    X_Train1 = SplitMCa["X_train"]
    X_Test1 = SplitMCa["X_test"]
    Y_Train1 = SplitMCa["Y_train"]
    Y_Test1 = SplitMCa["Y_test"]
    train_weights1 = SplitMCa["W_train"]
    test_weights1 = SplitMCa["W_test"]
    S_train1 = SplitMCa["S_train"]
    S_test1 = SplitMCa["S_test"]
    X_Train2 = SplitMCd["X_train"]
    X_Test2 = SplitMCd["X_test"]
    Y_Train2 = SplitMCd["Y_train"]
    Y_Test2 = SplitMCd["Y_test"]
    train_weights2 = SplitMCd["W_train"]
    test_weights2 = SplitMCd["W_test"]
    S_train2 = SplitMCd["S_train"]
    S_test2 = SplitMCd["S_test"]
    X_Train3 = SplitMCe["X_train"]
    X_Test3 = SplitMCe["X_test"]
    Y_Train3 = SplitMCe["Y_train"]
    Y_Test3 = SplitMCe["Y_test"]
    train_weights3 = SplitMCe["W_train"]
    test_weights3 = SplitMCe["W_test"]
    S_train3 = SplitMCe["S_train"]
    S_test3 = SplitMCe["S_test"]
    X_train = np.array([*X_Train1, *X_Train2, *X_Train3])
    X_test = np.array([*X_Test1, *X_Test2, *X_Test3])
    Y_train = np.array([*Y_Train1, *Y_Train2, *Y_Train3])
    Y_test = np.array([*Y_Test1, *Y_Test2, *Y_Test3])
    train_weights = np.array([*train_weights1, *train_weights2, *train_weights3])
    test_weights = np.array([*test_weights1, *test_weights2, *test_weights3])
    S_train = np.array([*S_train1, *S_train2, *S_train3])
    S_test = np.array([*S_test1, *S_test2, *S_test3])
    # MC sample data paths
    '''
    args.MCa_path = args.MCa
    args.MCb_path = args.MCb
    (MCa, MCb, MCa_spec, MCb_spec, MCa_weights, MCb_weights, maxObjCount)  = DSNNr.get_data(args)
    #(MCa, MCb, MCa_weights, MCb_weights, maxObjCount) = DSNNr.get_data(args)
    # MCa = np.nan_to_num(MCa)
    # MCb = np.nan_to_num(MCb)
    
    # # Hack Electrons and Muons masses
    # print("pid:{}".format(MCa[:,:,4]))
    # print("mass:{}".format(MCa[:,:,3]))
    # boolArrayMCa = np.logical_or(MCa[:,:,4]==math.fabs(11), MCa[:,:,4]==math.fabs(13))
    # # FilterIndex_MCa = np.where(boolArrayMCa == True)
    # print("original leptons masses")
    # print(MCa[:,:,3][boolArrayMCa])
    # MCa[:,:,3][boolArrayMCa] = 1.0
    # print("pid:{}, mass:{}".format(MCa[:,:,4][0][2], MCa[:,:,3][0][2]))
    # boolArrayMCb = np.logical_or(MCb[:,:,4]==math.fabs(11), MCb[:,:,4]==math.fabs(13))
    # # FilterIndex_MCb = np.where(boolArrayMCb == True)
    # MCb[:,:,3][boolArrayMCb] = 1.0
    # print("after mass:{}".format(MCa[:,:,3]))
    #inputOriFile = "CombinedMCade5050.npz"
    '''
    inputOriaFile = "SherpaMCa.hadd.npz"
    inputOribFile = "MadgaphMCa.hadd.npz"
    MCa = np.load(inputOriaFile)["MCa"]
    MCb = np.load(inputOribFile)["MCb"]
    MCa_weights = np.load(inputOriaFile)["MCa_weights"]
    MCb_weights = np.load(inputOribFile)["MCb_weights"]
    #MCa_spec = np.load(inputOriFile)["MCa_spectors"]
    #MCb_spec = np.load(inputOriFile)["MCb_spectors"]
    MCa_spec = np.load(inputOriaFile)["MCa_spec"]
    MCb_spec = np.load(inputOribFile)["MCb_spec"]
    '''
    MCa[MCa==-99]=1.5
    MCb[MCb==-99]=1.5
    
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
    #) = DSNNr.handle_data(args, MCa, MCb, MCa_weights, MCb_weights, MCa_spectors, MCb_spectors)
    #np.savez("CombinedSplitMCade5050_mask15_noLumi", X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, W_train=train_weights, W_test=test_weights, S_train=S_train, S_test=S_test)
    '''
    inputFile = "CombinedSplitMCade5050.npz"   
    #inputFile = "CombinedSplitMCade5050_NoLumi.npz"
    X_train = np.load(inputFile)["X_train"]
    X_test = np.load(inputFile)["X_test"]
    Y_train = np.load(inputFile)["Y_train"]
    Y_test = np.load(inputFile)["Y_test"]
    train_weights = np.load(inputFile)["W_train"]
    test_weights = np.load(inputFile)["W_test"]
    n_features = len(args.features.split(","))
    print('X_train:{}, shape:{}'.format(X_train, X_train.shape))
    print(X_test)
    print(train_weights)
    print(test_weights)
    '''
    
    #exit(1)
    #model = DSNNr.basic_model(args, n_features)
    n_features = len(args.features.split(","))
    model = DSNNr.DS_model(n_features) 
    checkpoint = ModelCheckpoint('./saved_models_2015nodes_20000_d020F03_MCaBBLumi_mk15_5050/'+ "/model-{epoch:02d}.ckpt",
                                 monitor='val_loss',
                                 verbose=2,
                                 save_freq='epoch',
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='min')
    # csvLogger = CSVLogger('./logs/' + args.global_name + '_loss.csv', separator = ",",append=False)
    csvLogger = CSVLogger("training_2015nodes_2000_d020F03_MCaBBLumi_mk15_5050.csv", separator=",", append=False)
    earlyStopping = EarlyStopping(monitor='val_loss', 
                                  min_delta=0, 
                                  patience=20, 
                                  verbose=1, 
                                  restore_best_weights=True)
    
    callbacks = [checkpoint, csvLogger, earlyStopping]
    # callbacks = [checkpoint, earlyStopping]
    
    # -----------
    # Train model
    # -----------
    start_train = time.time()      
    history = model.fit(X_train, Y_train,
                        epochs = 1000,
                        batch_size = 2000,
                        validation_data = (X_test, Y_test, test_weights),
                        class_weight=class_weights,
                        sample_weight=train_weights,
                        verbose = 1, 
                        callbacks = callbacks) # Train the model with the new callback
    end_train = time.time()
    #print('Y_test:',Y_test)
    print("Time consumed in training: ",end_train - start_train)
    ## Train model
    #earlystopping = EarlyStopping(patience=10, restore_best_weights=True)
    # csv_logger = CSVLogger("training.log", separator=",", append=False)
    
    #model.fit(
    #    np.asarray(X_train).astype('float32'),
    #    np.asarray(Y_train).astype('float32'),
    #    epochs=1000,
    #    batch_size=50000,
    #    validation_data=(np.asarray(X_test).astype('float32'), np.asarray(Y_test).astype('float32'),
    #                     np.asarray(test_weights).astype('float32')),
    #    sample_weight=np.asarray(train_weights).astype('float32'),
    #    callbacks=[earlystopping, csv_logger],
    #    verbose=2,
    #
    #)
    #
    #model.save("{}.h5".format(args.global_name))

    #Plot training accuracy, validation accuracy training loss, and validation loss w.r.t epochs
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # colors = cycler(color=['#EE6666', '#3388BB', '#9988DD','#EECC55', '#88BB44', '#FFBBBB'])
    # plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=True, prop_cycle=colors)

    # epochs = range(1, len(acc) + 1)
    # plt.plot(epochs, acc, 'bo', label='Training acc', color='#EE6666')
    # plt.plot(epochs, val_acc, 'b', label='Validation acc', color='#3388BB')
    # plt.title('Training and validation accuracy')
    # # plt.ylabel("loss")
    # plt.xlabel("epoch")
    # # plt.legend()
    # # plt.figure()
    
    # plt.plot(epochs, loss, 'bo', label='Training loss', color='#9988DD')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss', color='#EECC55')
    # plt.title('Training and validation loss')
    # plt.legend()

    # # plt.show()
    # plt.savefig("training.png", bbox_inches="tight", dpi=1200)
    

if __name__ == "__main__":
    args = DSNNr.handle_args()
    train(args)
