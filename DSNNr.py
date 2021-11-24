from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from scipy.stats import entropy, chisquare, wasserstein_distance

import numpy as np
import pandas as pd
import awkward as ak
# energyflow imports
import energyflow as ef
from energyflow.archs import PFN
from energyflow.utils import data_split, remap_pids, to_categorical

from operator import itemgetter
import matplotlib.pyplot as plt
import uproot
import argparse
import os
import math
import itertools
import operator
import time
# Root numpy and ROOT imports moved to top for efficiency
#   Including exception handling for python module imports
#import sys
#sys.argv.append( '-b-' ) 
global ROOTPlot
try: 
    from ROOT import TCanvas, TH1F, gROOT, gStyle, TLegend, TLorentzVector, TPad, TLine
    import ROOT as ROOT
    import root_numpy as rn
except ImportError: 
    ROOTPlot = False
else: 
    ROOTPlot = True


from array import array
from collections import defaultdict

import ctypes

from matplotlib import rc

rc("text", usetex=True)


def handle_args():
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument("--global_name", type=str, help="name for this run")
    parser.add_argument("--MCa", type=str, default="")
    parser.add_argument("--MCb", type=str, default="")
    parser.add_argument("--TreeName", type=str, default="Nominal")
    parser.add_argument("--features", type=str, default="dRBB")
    parser.add_argument("--spectators", type=str, default="")
    parser.add_argument("--weightFeature", type=str, default="")
    #parser.add_argument("--plot_features", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--nEvents", type=int, default=100000)
    parser.add_argument("--VerbosePlot", type=int, default=-1)
    parser.add_argument("--dropout", type=float, default=-99.0)

    args = parser.parse_args()

    return args
          
def getNumObj(MC, feat, objList):
    maxObjCount = 0
    for row in range(0, len(MC[feat][:,0])):
        maxObjCount = len(MC[feat][row, :]) if len(MC[feat][row, :]) > maxObjCount else maxObjCount
    return maxObjCount

def CheckMaxObjCount(features,MCa_tree,MCb_tree, ChunkDivMCa, ChunkDivMCb):
    delayed_MCa = []
    delayed_MCb = []
    maxObjCount = 0
    for index,feat in enumerate(features):
        for idx,  (chunk_arrayMCa,chunk_arrayMCb) in enumerate(itertools.zip_longest(MCa_tree.iterate(step_size=ChunkDivMCa), MCb_tree.iterate(step_size=ChunkDivMCb))):
        
            #################### Function 1 #####################
            # Get the array - jagged array using awkward
#             awk_feature_array_a = MCa[feat].array()
#             awk_feature_array_b = MCb[feat].array()
        #     awk_feature_array_a = MCa[feat]
        #     awk_feature_array_b = MCb[feat]
            # Awkward does not allow conversion of variable length arrays to numpy natively
            # so ascertain maximum value before padding
            maxObjCount_MCa = getNumObj(chunk_arrayMCa, feat)
            delayed_MCa.append(maxObjCount_MCa)
            maxObjCount_MCb = getNumObj(chunk_arrayMCb, feat)
            delayed_MCb.append(maxObjCount_MCb)
            
    maxOMCa = max(delayed_MCa)
    maxOMCb = max(delayed_MCb)
    if maxOMCa > maxOMCb:  
        maxObjCount =  maxOMCa
    else:
        maxObjCount =  maxOMCb
#             for row in range(0, len(MCa[feat][:,0])):
#                 maxObjCount = len(MCa[feat][row, :]) if len(MCa[feat][row, :]) > maxObjCount else maxObjCount
                
#             for row in range(0,len(MCb[feat][:,0])):
#                 maxObjCount = len(MCb[feat][row, :]) if len(MCb[feat][row, :]) > maxObjCount else maxObjCount
            #>>>>>>>>>>>>>>>>> Function 1 <<<<<<<<<<<<<<<<<<<<<<<<
    return maxObjCount

 
def preprocess_data(
    MCa_path,
    MCb_path,
    features,
    weightFeature,
    parameters,
    spectators,
    nEvents,
    tree_name="Nominal"
):
    # grab our data and iterate over chunks of it with uproot
    print("Uproot open file")

    MCa_file = uproot.open(MCa_path) 
    MCb_file = uproot.open(MCb_path)
    # Now get the Tree 
    print("Getting TTree from file")
    MCa_tree = MCa_file[tree_name]
    MCb_tree = MCb_file[tree_name]
    nMCaEvents = MCa_tree.num_entries
    nMCbEvents = MCb_tree.num_entries
    print("nMCaEvents:{}, nMCbEvents:{}".format(nMCaEvents, nMCbEvents))
    ChunkDivMCa = math.ceil(nMCaEvents/1)  # 50 is meant for having 50 chnuks.
    ChunkDivMCb = math.ceil(nMCbEvents/1)

    MCa_total = []
    MCb_total = []
    MCa_weights_total = []
    MCb_weights_total= []
    MC_weights_chunk=[]
    delayed_MC = []
    MC_weight = []
    MC_spec = []
    obj_list = ["el_pt","mu_pt","tau_pt"]
    maxObjs = len(obj_list)
    # Seperate the CheckMaxObjCount out from the PadFeatures as it needs to loop over the arrays anyway. We can do another chunking here and make things tidy.
    # maxObjs = CheckMaxObjCount(features, MCa_tree, MCb_tree, ChunkDivMCa, ChunkDivMCb)
    # ChunkDiv=int(nEvents/10)
    for idx,  (chunk_arrayMCa,chunk_arrayMCb) in enumerate(itertools.zip_longest(MCa_tree.iterate(step_size=ChunkDivMCa), MCb_tree.iterate(step_size=ChunkDivMCb))):
        # Padding arrays
        dfs_delayed_MC= PadFeatures(chunk_arrayMCa, chunk_arrayMCb, features, maxObjs)
        delayed_MC.append(dfs_delayed_MC)

        # Now check if the user specified a weight feature to be used
        #  -> I.e. weighted events otherwise set all weights to 1.0
        # MCa_weights,MCb_weights = AddDataPointWeights(MCa_data, MCb_data, features, weightFeature, len(MCa[:,0,0]), len(MCb[:,0,0]))
        MC_weights_chunk = AddDataPointWeights(chunk_arrayMCa , chunk_arrayMCb, features, weightFeature)
        MC_spec_chunk = FormSpectators(chunk_arrayMCa, chunk_arrayMCb, spectators, len(chunk_arrayMCa), len(chunk_arrayMCb))
        MC_spec.append(MC_spec_chunk)
        # Induce a shape difference just to test the code base
        #MCb_weights = InduceShapeDiff(MCb, MCb_weights, features)
        
        # Scale weights to 1pb^{-1}
        #MC_weights_chunk = LumiScale(MC_weights_chunk[0], MC_weights_chunk[1], MCa_path, MCb_path, weightFeature, tree_name, MCa_tree, MCb_tree, features)
        MC_weight.append(MC_weights_chunk)

        # Concatenate array
    MCa_total = np.concatenate([delayed_MC[item][0] for item in range(len(delayed_MC))]) 
    MCb_total = np.concatenate([delayed_MC[item][1] for item in range(len(delayed_MC))]) 
    MCa_weights_total = np.concatenate([MC_weight[item][0] for item in range(len(MC_weight))]) 
    MCb_weights_total = np.concatenate([MC_weight[item][1] for item in range(len(MC_weight))]) 
    MCa_spec = np.concatenate([MC_spec[item][0] for item in range(len(MC_spec))]) 
    MCb_spec = np.concatenate([MC_spec[item][1] for item in range(len(MC_spec))])
    
    #if len(MCa_total[:,0,0]) > nEvents and len(MCb_total[:,0,0]) > nEvents:
    print("    Length total : \n {}".format(len(MCa_total[:,0,0])))
    print("    Length weightsa total : \n {}".format(len(MCa_weights_total)))
    print("    Length total: \n {}".format(len(MCb_total[:,0,0])))
    print("    Length weightsb total : \n {}".format(len(MCb_weights_total)))
                

    MCa_total, MCb_total = PreScaleInputRange(MCa_total, MCb_total, features, "linear")
   
    # return (MCa_total, MCb_total, MCa_weights_total, MCb_weights_total, maxObjs)

    return (MCa_total, MCb_total, MCa_spec, MCb_spec, MCa_weights_total, MCb_weights_total, maxObjs)

def get_data(args, getnorm=False):
    # Check if features were loaded
    if args.features:
        my_features = args.features.split(",")
    else:
        my_features = [
            "pt",
            "eta",
            "phi",
            "m",
            "pdgid",
        ]
    # Check if spectators were loaded
    if args.spectators:
        my_spectators = args.spectators.split(",")
    else:
        my_spectators = []

    # Parameters
    my_parameters = []
    print("Spec:{}".format(my_spectators))

    # Get the name of the root files
    file_MCa = args.MCa.split('/')[-1]
    file_MCa = file_MCa.replace(".root", ".npz")
    file_MCb = args.MCb.split('/')[-1]
    file_MCb = file_MCb.replace(".root", ".npz")

    print("get_data:  Loading data")
    print(os.getcwd())
    print(file_MCa)
    print(file_MCb)
    start = time.time()
    if os.path.isfile(file_MCa) and os.path.isfile(file_MCb):
        print("get_data:  Loading data from npz")
        MCa = np.load(file_MCa)['MCa']
        MCb = np.load(file_MCb)['MCb']
        MCa_spec = np.load(file_MCa)['MCa_spec']
        MCb_spec = np.load(file_MCb)['MCb_spec']
        MCa_weights = np.load(file_MCa, allow_pickle=True)['MCa_weights']
        MCb_weights = np.load(file_MCb, allow_pickle=True)['MCb_weights']
        nEvents,maxObjCount,nFeatures = np.shape(MCa)
    else:
        print("get_data:  Loading data from root file")
        (
            # MCa, MCb, MCa_weights, MCb_weights, maxObjCount
            MCa, MCb, MCa_spec, MCb_spec, MCa_weights, MCb_weights, maxObjCount
        ) = preprocess_data(
            args.MCa,
            args.MCb,
            my_features,
            args.weightFeature,
            my_parameters,
            my_spectators,
            args.nEvents,
            args.TreeName,
        )
    end = time.time()
    print("Time consumed in working: ",end - start)
    print("len(MCa):{}, len(MCb):{}".format(len(MCa[1]), len(MCb[1])))

    ### adding PdgID artificially
    MCa_input = []
    MCb_input = []
    # obj_list = [11,13,15,510,520,100,42]
    is5Dim=False
    if is5Dim:
        obj_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
        for row in range(0, len(MCa)):
            for obj_idx,obj in enumerate(obj_list):
                MCa_input.append(np.append(MCa[row][obj_idx], obj))
        MCa_input =  np.reshape(MCa_input, (len(MCa), len(obj_list), 5))
        for row in range(0, len(MCb)):
            for obj_idx,obj in enumerate(obj_list):
                MCb_input.append(np.append(MCb[row][obj_idx], obj))
        MCb_input =  np.reshape(MCb_input, (len(MCb), len(obj_list), 5))
        #####
        # Save numpy arrays 
        # np.savez(file_MCa, MCa=MCa_input, MCa_weights=MCa_weights)
        # np.savez(file_MCb, MCb=MCb_input, MCb_weights=MCb_weights)
        print("Fianl MCa shape:{}, spec shape:{}".format(np.array(MCa_input).shape, np.array(MCa_spec).shape))
        np.savez(file_MCa, MCa=MCa_input, MCa_weights=MCa_weights, MCa_spec=MCa_spec)
        np.savez(file_MCb, MCb=MCb_input, MCb_weights=MCb_weights, MCb_spec=MCb_spec)
    else:
        np.savez(file_MCa, MCa=MCa, MCa_weights=MCa_weights, MCa_spec=MCa_spec)
        np.savez(file_MCb, MCb=MCb, MCb_weights=MCb_weights, MCb_spec=MCb_spec)
    # Print if requested plot features
    
    #if ROOTPlot:   # Only plot if ROOT is installed correctly with python enabled binaries
        #FeaturePlotter  (MCa, MCb, MCa_weights, MCb_weights, maxObjCount, my_features, args)
    #    SpectatorPlotter(MCa_spec, MCb_spec, MCa_weights, MCb_weights, my_spectators, args)
    if is5Dim:
        return (MCa_input, MCb_input, MCa_spec, MCb_spec, MCa_weights, MCb_weights, maxObjCount) 
    else:
        return (MCa, MCb, MCa_spec, MCb_spec, MCa_weights, MCb_weights, maxObjCount) 
    # return (MCa_input, MCb_input, MCa_weights, MCb_weights, maxObjCount)


def handle_data(args, MCa, MCb, MCa_weights, MCb_weights, MCa_spec, MCb_spec):
    MCa_labels = to_categorical(np.zeros(MCa.shape[0]), num_classes=2)
    MCb_labels = to_categorical(np.ones(MCb.shape[0]), num_classes=2)
    X = np.concatenate( (MCa, MCb) )
    Y = np.concatenate( (MCa_labels, MCb_labels) )
    W = np.concatenate( (MCa_weights, MCb_weights) )
    S = np.concatenate( (MCa_spec, MCb_spec) ) 
    Y_class = argmax(Y, axis=1)
    unique_y = [1.,0.] #1 for the MCa, 0 for the MCb
    unique_classes = np.unique(Y_class)
    class_weights = dict(zip(unique_y,sklearn.utils.class_weight.compute_class_weight('balanced', unique_classes, Y_class)))

    X_train, X_test, Y_train, Y_test, W_train, W_test, S_train, S_test = train_test_split(X, Y, W, S, test_size=0.5, shuffle=True,random_state=1)

    #Store the numpy to disk
    np.savez("DataMCaSplit5050_MCaBB_split", X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, W_train=W_train, W_test=W_test, S_train=S_train, S_test=S_test)

    return X_train, X_test, Y_train, Y_test, W_train, W_test, S_train, S_test, class_weights


def basic_model(args, n_features=1):

    inputs = Input((n_features,))

    hidden_layer_1 = Dense(
        args.width, kernel_initializer="lecun_normal", activation="selu"
    )(inputs)
    dropout_layer_1 = Dropout(args.dropout)(hidden_layer_1)
    hidden_layer_2 = Dense(
        args.width, kernel_initializer="lecun_normal", activation="selu"
    )(dropout_layer_1 if args.dropout != -99.0 else hidden_layer_1)
    dropout_layer_2 = Dropout(args.dropout)(hidden_layer_2)
    hidden_layer_3 = Dense(
        args.width, kernel_initializer="lecun_normal", activation="selu"
    )(dropout_layer_2 if args.dropout != -99.0 else hidden_layer_2)
    dropout_layer_3 = Dropout(args.dropout)(hidden_layer_3)

    outputs = Dense(2, activation="softmax")(
        dropout_layer_3 if args.dropout != -99.0 else hidden_layer_3
    )

    opt = Adam(lr=args.lr)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        weighted_metrics=["categorical_crossentropy"],
    )

    return model

def DS_model(in_dim):

    # network architecture parameters
    Phi_sizes = (80,80,60) #(140,140,168)
    F_sizes = (80,80,80) #(140,140,140)
    ##https://github.com/keras-team/keras/blob/68dc181a5e34d1f20edabe531176b3bfb50001f9/keras/engine/training.py#L382-L383
    ##metrics: List of metrics to be evaluated by the model during training and testing.
    ##weighted_metrics: List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing
    compile_opts={ 'loss':'categorical_crossentropy',
                   #'optimizer':'adam',
                   #'metrics':'acc',
                   'optimizer':'adamax',
                   'metrics':'categorical_accuracy',
                   'weighted_metrics':["categorical_crossentropy"],
                  }
    print("in_dim:{}".format(in_dim))
    DSNNr = PFN(input_dim=in_dim, 
                Phi_sizes=Phi_sizes, F_sizes=F_sizes,
                summary=False,
                mask_val=1.5,
                #Phi_acts='softmax',
                #F_acts='softmax',
                #dense_dropouts=0.25,
                latent_dropout=0.2,
                F_dropouts=0.3,
                compile_opts=compile_opts)

    return DSNNr


def eval_model(model, X_input):
    preds = model.predict(X_input)
    weights = preds[:, 1] / preds[:, 0]
    return weights


#def PadFeatures(MCa, MCb, features, nEventsA, nEventsB, PadFeatures):
def PadFeatures(MCa, MCb, features, nObjsPerFeature):
    flag_reweightB1 = False
    # Local scope storage of total number of features
    #   -> Use the first feature as an example of how big to make the 
    #   -> numpy array
#     nObjsPerFeature =-1
    # Get the array - jagged array using awkward
    len_feature = len(features)
    ele_4vector = np.array(["el_pt", "el_eta","el_phi","el_m","el_pdgid"])

    muon_4vector = np.array(["mu_pt", "mu_eta","mu_phi","mu_m", "mu_pdgid"])

    # tau_4vector = np.array(["tau_pt", "tau_eta", "tau_phi", "tau_m"])
    # n_tau_4vector = np.array(["n_tau_pt", "n_tau_eta", "n_tau_phi", "n_tau_m"])
    # p_tau_4vector = np.array(["p_tau_pt", "p_tau_eta", "p_tau_phi", "p_tau_m"])

    B1_4vector = np.array(["pTB1","etaB1","phiB1", "mB1","pdgIDB1"])
    B2_4vector = np.array(["pTB2", "etaB2", "phiB1", "mB1", "pdgIDB2"])
    # pfjet_4vector = np.array(["pfjet_pt","pfjet_eta","pfjet_phi","pfjet_m"])
    thirdjet_4vector = np.array(["pTJ3","etaJ3","phiJ3","mJ3", "pdgIDJ3"]) #the third jets
    met_4vector = np.array(["met_met", "met_eta", "met_phi", "met_m", "met_pdgid"])

    # obj_pt  = np.array(["el_pt", "mu_pt"])
    # obj_eta = np.array(["el_eta", "mu_eta"])
    # obj_phi = np.array(["el_phi", "mu_phi"])
    # obj_m   = np.array(["el_m", "mu_m"])
    
    #create feature tuples for an object 
    a_ele = [MCa[ele_4vector[i]] for i in range(len(ele_4vector))]
    b_ele = [MCb[ele_4vector[i]] for i in range(len(ele_4vector))]

    a_muon = [MCa[muon_4vector[i]] for i in range(len(muon_4vector))]
    b_muon = [MCb[muon_4vector[i]] for i in range(len(muon_4vector))]
    a_B1 = [MCa[B1_4vector[i]] for i in range(len(B1_4vector))]
    if flag_reweightB1:
        a_B1[0] = a_B1[0]+a_B1[0]/100
    b_B1 = [MCb[B1_4vector[i]] for i in range(len(B1_4vector))]
    a_B2 = [MCa[B2_4vector[i]] for i in range(len(B2_4vector))]
    b_B2 = [MCb[B2_4vector[i]] for i in range(len(B2_4vector))]
    # a_pfjet = [MCa[pfjet_4vector[i]] for i in range(len(pfjet_4vector))]
    # b_pfjet = [MCb[pfjet_4vector[i]] for i in range(len(pfjet_4vector))]
    a_thirdjet = [MCa[thirdjet_4vector[i]] for i in range(len(thirdjet_4vector))]
    b_thirdjet = [MCb[thirdjet_4vector[i]] for i in range(len(thirdjet_4vector))]

    a_met = [MCa[met_4vector[i]] for i in range(len(met_4vector))]
    b_met = [MCb[met_4vector[i]] for i in range(len(met_4vector))]

    ele_MCa_events = zip(*a_ele)
    ele_MCb_events = zip(*b_ele)

    mu_MCa_events = zip(*a_muon)
    mu_MCb_events = zip(*b_muon)

    B1_MCa_events = zip(*a_B1)
    B2_MCa_events = zip(*a_B2)
    B1_MCb_events = zip(*b_B1)
    B2_MCb_events = zip(*b_B2)
    # pfjet_MCa_events = zip(*a_pfjet)
    # pfjet_MCb_events = zip(*b_pfjet)
    thirdjet_MCa_events = zip(*a_thirdjet)
    thirdjet_MCb_events = zip(*b_thirdjet)

    met_MCa_events = zip(*a_met)
    met_MCb_events = zip(*b_met)
    MCa_Events = list(zip(ele_MCa_events, mu_MCa_events, B1_MCa_events, B2_MCa_events, thirdjet_MCa_events, met_MCa_events))
    MCb_Events = list(zip(ele_MCb_events, mu_MCb_events, B1_MCb_events, B2_MCb_events, thirdjet_MCb_events, met_MCb_events))
    print("Event MCa shape:{}".format(np.array(list(MCa_Events)).shape))
    print("Event MCb shape:{}".format(np.array(list(MCb_Events)).shape))
    print("MCa:{}".format(np.array(list(MCb_Events))))
    outputs = []
    outputs = [np.array(list(MCa_Events)), np.array(list(MCb_Events))]
    return outputs

def FormSpectators(MCa, MCb, spectators, nEventsA, nEventsB):

    # Convert awkward array of spectators into numpy arrays
    MCa_spec_total = np.full( (nEventsA, len(spectators)), np.nan )
    MCb_spec_total = np.full( (nEventsB, len(spectators)), np.nan )
    for index,spec in enumerate(spectators):
        # Get awkward array from original uproot dataset
        #awk_array_a = MCa[spec].array()
        awk_array_a = MCa[spec]
        #awk_array_b = MCb[spec].array()
        awk_array_b = MCb[spec]

        # Now convert to numpy array and store in storage array
        MCa_spec_total[:,index] = ak.to_numpy(awk_array_a[:nEventsA])
        MCb_spec_total[:,index] = ak.to_numpy(awk_array_b[:nEventsB])

    results=[]
    results=[MCa_spec_total,MCb_spec_total]
        
    return results
    

def AddDataPointWeights(MCa, MCb, features, weightFeature):
    outputs=[]
    weightFeature="EventWeight"
    if weightFeature != "":
        # Extract the awkward array from the TTree
        weights_a = MCa[weightFeature]
        weights_b = MCb[weightFeature]
        # Convert to numpy
        np_MCa = ak.to_numpy( weights_a )
        np_MCb = ak.to_numpy( weights_b )
        # return the numpy arrays of the weights
        outputs=[np_MCa, np_MCb]
        return outputs
        return np_MCa, np_MCb
    
    # print("MCa shape in weights:{}".format(np.array(MCa).shape))
    # if weightFeature != "":
    #     weights_a = MCa[weightFeature]
    #     weights_b = MCb[weightFeature]
    #     np_MCa = np.ones([2,len(weights_a),4])
    #     np_MCb = np.ones([2,len(weights_b),4])
    #     print("weights_a:{}, len:{}".format(weights_a, len(weights_a)))
    #     for row, weight in enumerate(weights_a):
    #         np_MCa[:, row, :] = np.array(weight)
    #     for row, weight in enumerate(weights_b):
    #         np_MCb[:, row, :] = np.array(weight)
    #     outputs=[ np_MCa, np_MCb]
    #     return outputs
    else:
        # return the numpy arrays of the weights with all weights of 1.0
        np_MCa = np.ones( shape = ( len( MCa ) ) )
        np_MCb = np.ones( shape = ( len( MCb ) ) )
        outputs=[ np_MCa, np_MCb]
        return outputs
#         return np_MCa, np_MCb

def FeaturePlotter(MCa, MCb, weightsa, weightsb, maxObjCount, features, args):

    ROOT.PyConfig.IgnoreCommandLineOptions = True  # stop PyRoot hijacking -h WHY DOESNT THIS ALWAYS WORK
    ROOT.gROOT.SetBatch(True)  # Don't want to plot these to screen as we generate them

    print("FeaturePlotter:  len(MCa[:,0,0]) = {}".format( len(MCa[:,0,0])) )
    print("FeaturePlotter:  weightsa        = {}".format( len(weightsa) ) )

    # Remove stat box
    gStyle.SetOptStat(0)
    gStyle.SetTitleFontSize(0.08)
    gStyle.SetLabelSize(0.07)
    # Get the index of the pdgid feature
    pdgid_index = features.index('TruthObj_pdgid')

    # Get the number of unique pdgid elements
    #   -> Consider the first 5% of the sample only for speed
    pdgid_a = MCa[:,:,pdgid_index]
    pdgid_b = MCb[:,:,pdgid_index]
    #flattenedPDGID = pdgid_a.flatten()
    #pdgids = []
    pdgids = [-11,11,-13,13,-15,15,0,4,5,42,100,200,204,205] #pdg 0 is broken as I include a dummy slice of 0's in MCa/b FIX!!!
    #pdgids = [0, 2/115, 4/115, 15/115, 19/115, 20/115, 26/115, 28/115, 30/115, 57/115, 115/115] #pdg 0 is broken as I include a dummy slice of 0's in MCa/b FIX!!!
    #for elem in range( len( flattenedPDGID ) ):
    #    print(elem)
    #    if not pdgids.count(flattenedPDGID[elem]):
    #        pdgids.append(flattenedPDGID[elem])
    
    # Loop through features
    for index, feat in enumerate(features):
        for pdg in pdgids:
            print("Forming Feature Plots for:  PDG {},  feature: {} ".format(pdg,feat))

            # Extract the feature slices
            featSlice_a = MCa[:, :, index]
            featSlice_b = MCb[:, :, index]

            #pdgID filter arrays
            filter_a = pdgid_a==pdg
            filter_b = pdgid_b==pdg

            # Request only a specific pdg ID
            featSlice_a_cut = featSlice_a[filter_a]
            featSlice_b_cut = featSlice_b[filter_b]

            # Flatten all objs with same pdgid into a 1D array
            featSlice_a_flat = featSlice_a_cut.flatten()
            featSlice_b_flat = featSlice_b_cut.flatten()
            
            # Bail if there are no events
            if len(featSlice_a_flat) == 0 or len(featSlice_b_flat) == 0:
                continue

            # Ascertain maximum and minimum
            minElem = np.amin(featSlice_a_flat)
            maxElem = np.amax(featSlice_a_flat)
            # Number of divisions
            nDiv = 100
            
            # Create the weight array of the same shape as the featSlice_... before flattening
            weightsa_expand = np.repeat( np.expand_dims(weightsa, axis=1), repeats=maxObjCount, axis = 1  )
            weightsb_expand = np.repeat( np.expand_dims(weightsb, axis=1), repeats=maxObjCount, axis = 1  )

            # Create a canvas
            can1 = TCanvas(feat+"_"+str(pdg), feat+"_"+str(pdg), 1200,800)
            mainPad =  TPad("mainPad", "top",    0.0, 0.37, 1.0, 1.00)
            ratioPad = TPad("ratioPad","bottom", 0.0, 0.02, 1.0, 0.43)
            mainPad.SetBottomMargin(0.38) 
            mainPad.Draw()
            ratioPad.Draw()
            
            
            # Create two histograms
            mainPad.cd()
            hista = TH1F(feat+"_a", feat+"_a", nDiv, minElem, maxElem)
            histb = TH1F(feat+"_b", feat+"_b", nDiv, minElem, maxElem)
            
            # Fill with vector
            print("featSlice_a_flat, {}, size: {}".format(featSlice_a_flat, len(featSlice_a_flat)))
            print("featSlice_b_flat, {}, size: {}".format(featSlice_b_flat, len(featSlice_b_flat)))
            # print("featSlice_a_flat size:  {}".format(len(featSlice_a_flat)))
            # print("featSlice_b_flat size:  {}".format(len(featSlice_b_flat)))
            print("weightsa, {}, size{}".format(weightsa[filter_a.any(axis=1)], len(weightsa_expand[filter_a])))
            print("weightsb, {}, size{}".format(weightsb[filter_b.any(axis=1)], len(weightsb_expand[filter_b])))
            #print("weightsa size:  {}".format(len(weightsa[filter_a.any(axis=1)])))
            # print("weightsa 2D size:  {}".format(len(weightsa_expand[filter_a])))
            #print("weightsb size:  {}".format(len(weightsb[filter_b.any(axis=1)])))
            # print("weightsb 2D size:  {}".format(len(weightsb_expand[filter_b])))
            #rn.fill_hist(hista, featSlice_a_flat, weightsa[filter_a.any(axis=1)])
            #rn.fill_hist(histb, featSlice_b_flat, weightsb[filter_b.any(axis=1)])
            rn.fill_hist(hista, featSlice_a_flat, weightsa_expand[filter_a])
            rn.fill_hist(histb, featSlice_b_flat, weightsb_expand[filter_b])
            
            # Now define the TLegend
            legend = TLegend(0.7,0.65,0.85,0.85)
            legend.SetBorderSize(0)
            #legend.SetTextSize(0)
            legend.AddEntry(hista, "MC_{a}", "f")
            legend.AddEntry(histb, "MC_{b}", "f")
            legend.SetTextSize(0.05)

            # Draw histograms
            #    MCa
            hista.GetYaxis().SetTitle("#frac{d #sigma}{d"+feat+"}")
            # hista.GetXaxis().SetTitle(feat)
            hista.SetFillStyle(1001)
            hista.SetFillColorAlpha(4, 0.25);
            hista.SetLineColor(4)
            hista.DrawCopy("hist")

            #    MCb
            histb.GetYaxis().SetTitle("#frac{d#sigma}{d"+feat+"}")
            # histb.GetXaxis().SetTitle(feat)
            histb.SetFillStyle(1001)
            histb.SetFillColorAlpha(2, 0.25);
            histb.SetLineColor(2)
            histb.DrawCopy("hist SAME")

            # Draw legend
            legend.Draw()
            # Draw Ratio Plot
            # can1.cd().SetLogy()

            ratioPad.cd()
            ratio_baseline = hista.Clone("ratio_hist_"+feat)
            ratio_baseline.Divide(histb)
            ratio_baseline.SetFillColorAlpha(0, 0.25);
            #TLine
            line = TLine(ratio_baseline.GetXaxis().GetXmin(),1,ratio_baseline.GetXaxis().GetXmax(),1)
            line.SetLineWidth(2)
            line.SetLineStyle(2)
            # ratio_baseline.SetLineWidth(1)
            # ratio_baseline.SetLineColor(14)
            ratio_baseline.SetTitle("")
            ratio_baseline.GetYaxis().SetTitle("w.r.t MCb")
            ratio_baseline.GetYaxis().SetRangeUser(0.8,1.2)
            ratio_baseline.GetYaxis().CenterTitle(1)
            ratio_baseline.GetYaxis().SetTitleOffset(0.3)
            ratio_baseline.GetYaxis().SetTitleSize(0.12)
            ratio_baseline.GetYaxis().SetLabelSize(0.07)
            ratio_baseline.GetXaxis().SetTitle(feat)
            ratio_baseline.GetXaxis().SetTitleOffset(0.65)
            ratio_baseline.GetXaxis().SetTitleSize(0.2)
            #ratio_baseline.GetXaxis().SetLabelSize(0.15)
            ratio_baseline.Draw("E0")
            line.Draw("SAME")

            # Save the Canvas
            cwd = os.getcwd()+"/"
            can1.SaveAs(cwd+feat+"_"+str(pdg)+".png")
            can1.SaveAs(cwd+feat+"_"+str(pdg)+".pdf")
            can1.SaveAs(cwd+feat+"_"+str(pdg)+".eps")
            can1.SaveAs(cwd+feat+"_"+str(pdg)+".root")

def SpectatorPlotter(MCa, MCb, weightsa, weightsb, spectators, args):

    ROOT.PyConfig.IgnoreCommandLineOptions = True  # stop PyRoot hijacking -h WHY DOESNT THIS ALWAYS WORK
    ROOT.gROOT.SetBatch(True)  # Don't want to plot these to screen as we generate them

    # Remove stat box
    gStyle.SetOptStat(0)
    gStyle.SetTitleFontSize(0.08)
    gStyle.SetLabelSize(0.07)
    
    # Loop through features
    for index, spec in enumerate(spectators):
        print("Forming Spectators Plots for:  {} ".format(spec))

        # Extract the feature slices
        specSlice_a = MCa[:, index]
        specSlice_b = MCb[:, index]

        # Flatten all objs with same pdgid into a 1D array
        specSlice_a_flat = specSlice_a.flatten()
        specSlice_b_flat = specSlice_b.flatten()
            
        # Bail if there are no events
        if len(specSlice_a_flat) == 0 or len(specSlice_b_flat) == 0:
            continue

        # Ascertain maximum and minimum
        minElem = np.amin(specSlice_a_flat)
        maxElem = np.amax(specSlice_a_flat)
        # Number of divisions
        nDiv = 100
            
        # Create a canvas
        can1 = TCanvas(spec, spec, 1200,800)
        mainPad =  TPad("mainPad", "top",    0.0, 0.37, 1.0, 1.00)
        ratioPad = TPad("ratioPad","bottom", 0.0, 0.02, 1.0, 0.43)
        #mainPad.SetBottomMargin(0.38) 
        ratioPad.SetTopMargin(0.07) 
        #ratioPad.SetBottomMargin(0.05) 
        mainPad.Draw()
        ratioPad.Draw()
                    
        # Create two histograms
        mainPad.cd()
        hista = TH1F(spec+"_a", spec+"_a", nDiv, minElem, maxElem)
        histb = TH1F(spec+"_b", spec+"_b", nDiv, minElem, maxElem)
            
        # Fill with vector
        #print("specSlice_a_flat, {}, size: {}".format(specSlice_a_flat, len(specSlice_a_flat)))
        #print("specSlice_b_flat, {}, size: {}".format(specSlice_b_flat, len(specSlice_b_flat)))
        # print("specSlice_a_flat size:  {}".format(len(specSlice_a_flat)))
        # print("specSlice_b_flat size:  {}".format(len(specSlice_b_flat)))
        #print("weightsa, {}, size{}".format(weightsa[filter_a.any(axis=1)], len(weightsa_expand[filter_a])))
        #print("weightsb, {}, size{}".format(weightsb[filter_b.any(axis=1)], len(weightsb_expand[filter_b])))
        #print("weightsa size:  {}".format(len(weightsa[filter_a.any(axis=1)])))
        # print("weightsa 2D size:  {}".format(len(weightsa_expand[filter_a])))
        #print("weightsb size:  {}".format(len(weightsb[filter_b.any(axis=1)])))
        # print("weightsb 2D size:  {}".format(len(weightsb_expand[filter_b])))
        #rn.fill_hist(hista, specSlice_a_flat, weightsa[filter_a.any(axis=1)])
        #rn.fill_hist(histb, specSlice_b_flat, weightsb[filter_b.any(axis=1)])
        rn.fill_hist(hista, specSlice_a_flat, weightsa)
        rn.fill_hist(histb, specSlice_b_flat, weightsb)
            
        # Now define the TLegend
        legend = TLegend(0.7,0.65,0.85,0.85)
        legend.SetBorderSize(0)
        #legend.SetTextSize(0)
        legend.AddEntry(hista, "MC_{a}", "f")
        legend.AddEntry(histb, "MC_{b}", "f")
        legend.SetTextSize(0.05)

        # Draw histograms
        #    MCa
        hista.GetYaxis().SetTitle("#frac{d #sigma}{d"+spec+"}")
        # hista.GetXaxis().SetTitle(spec)
        hista.SetFillStyle(1001)
        hista.SetFillColorAlpha(4, 0.25);
        hista.SetLineColor(4)
        hista.DrawCopy("hist")
        
        #    MCb
        histb.GetYaxis().SetTitle("#frac{d#sigma}{d"+spec+"}")
        # histb.GetXaxis().SetTitle(spec)
        histb.SetFillStyle(1001)
        histb.SetFillColorAlpha(2, 0.25);
        histb.SetLineColor(2)
        histb.DrawCopy("hist SAME")
        
        # Draw legend
        legend.Draw()
        # Draw Ratio Plot
        # can1.cd().SetLogy()
        
        ratioPad.cd()
        ratio_baseline = hista.Clone("ratio_hist_"+spec)
        ratio_baseline.Divide(histb)
        ratio_baseline.SetFillColorAlpha(0, 0.25);
        #TLine
        line = TLine(ratio_baseline.GetXaxis().GetXmin(),1,ratio_baseline.GetXaxis().GetXmax(),1)
        line.SetLineWidth(2)
        line.SetLineStyle(2)
        # ratio_baseline.SetLineWidth(1)
        # ratio_baseline.SetLineColor(14)
        ratio_baseline.SetTitle("")
        ratio_baseline.GetYaxis().SetTitle("w.r.t MCb")
        ratio_baseline.GetYaxis().SetRangeUser(0.8,1.2)
        ratio_baseline.GetYaxis().CenterTitle(1)
        ratio_baseline.GetYaxis().SetTitleOffset(0.3)
        ratio_baseline.GetYaxis().SetTitleSize(0.12)
        ratio_baseline.GetYaxis().SetLabelSize(0.07)
        ratio_baseline.GetXaxis().SetTitle(spec)
        ratio_baseline.GetXaxis().SetTitleOffset(0.65)
        ratio_baseline.GetXaxis().SetTitleSize(0.2)
        #ratio_baseline.GetXaxis().SetLabelSize(0.15)
        ratio_baseline.Draw("E0")
        line.Draw("SAME")
        
        # Save the Canvas
        cwd = os.getcwd()+"/"
        can1.SaveAs(cwd+spec+".png")
        can1.SaveAs(cwd+spec+".pdf")
        can1.SaveAs(cwd+spec+".eps")
        can1.SaveAs(cwd+spec+".root")
            

# Normalise features - standard scaling for now
def  PreScaleInputRange(MCa, MCb, features, type="linear"):
    # Extract maximum and minimum value of each 
    max = []
    min = []
    for index,feat in enumerate(features):
        # # skip pdgid
        if feat == "pdgid":
            continue

        # feature_array_a = MCa[:,:,index].flatten()
        # feature_array_b = MCb[:,:,index].flatten()

        filter_a = MCa[:, :, index] > -5
        filter_b = MCb[:, :, index] > -5
        array_a = MCa[:,:,index][filter_a]
        array_b = MCb[:,:,index][filter_b]

        feature_array_a = array_a.flatten()
        feature_array_b = array_b.flatten()
        # feature_array_a = np.ma.masked_equal(feature_array_a, -99, copy=False)
        # feature_array_b = np.ma.masked_equal(feature_array_b, -99, copy=False)
        # Now pre-process the range
        if type == "linear":

            # Now search for min and max
            # max_a = np.amax(feature_array_a)
            # max_b = np.amax(feature_array_b)
            # min_a = np.amin(feature_array_a)
            # min_b = np.amin(feature_array_b)
            max_a = np.nanmax(feature_array_a)
            max_b = np.nanmax(feature_array_b)
            min_a = np.nanmin(feature_array_a)
            min_b = np.nanmin(feature_array_b)

            max.append(max_a if max_a > max_b else max_b)
            min.append(min_a if min_a < min_b else min_b)

            print("PreScaleInputs::  feature = {}",format(feat))
            print("PreScaleInputs::      max = {}",format(max[-1]))
            print("PreScaleInputs::      min = {}",format(min[-1]))

            #print("PreScaleInputRange: MCa(preScale)   {}, size:{}".format(MCa[:,:,index], len(MCa)))
            #print("PreScaleInputRange: MCb(preScale)   {}, size:{}".format(MCb[:,:,index], len(MCb)))

            MCa[:,:,index] =MCa[:,:,index]-min[-1]
            MCa[:,:,index] =MCa[:,:,index]/(max[-1]-min[-1])

            MCb[:,:,index] =MCb[:,:,index]-min[-1]
            MCb[:,:,index] =MCb[:,:,index]/(max[-1]-min[-1])

            print("PreScaleInputs: MCa   {}, size:{}".format(MCa[:,:,index], len(MCa)))
            print("PreScaleInputs: MCb   {}, size:{}".format(MCb[:,:,index], len(MCb)))

        if type == "stdandard":

            # Now search for the mean
            mean_a = np.nanmean(feature_array_a)
            mean_b = np.nanmean(feature_array_b)
            std_a = np.nanstd(feature_array_a)
            std_b = np.nanstd(feature_array_b)

            #max.append(max_a if max_a > max_b else max_b)
            #min.append(min_a if min_a < min_b else min_b)

            #print("PreScaleInputs::  feature = {}",format(feat))
            #print("PreScaleInputs::      max = {}",format(max[-1]))
            #print("PreScaleInputs::      min = {}",format(min[-1]))

            #print("PreScaleInputRange: MCa(preScale)   {}, size:{}".format(MCa[:,:,index], len(MCa)))
            #print("PreScaleInputRange: MCb(preScale)   {}, size:{}".format(MCb[:,:,index], len(MCb)))

            MCa[:,:,index] =MCa[:,:,index] - mean_a
            MCa[:,:,index] =MCa[:,:,index] / std_a

            MCb[:,:,index] =MCb[:,:,index] - mean_b
            MCb[:,:,index] =MCb[:,:,index] / std_b

            print("PreScaleInputs: MCa   {}, size:{}".format(MCa[:,:,index], len(MCa)))
            print("PreScaleInputs: MCb   {}, size:{}".format(MCb[:,:,index], len(MCb)))

    MCa = np.nan_to_num(MCa, nan=1.5)
    MCb = np.nan_to_num(MCb, nan=1.5)
    return MCa,MCb


# Determine sum of weights and then weight according to
# sigma/SumWeights
def LumiScale(weightsa, weightsb, MCa_path, MCb_path, weightFeature, tree_name, MCa, MCb, features):
    isAddSample = False #remember to fix it.
    isShapeNormReweight = False 
#     print("Uproot open file")
    print("weight shape in lumi:{}, weightb:{}".format(np.array(weightsa).shape, np.array(weightsb).shape))
    MCa_file = uproot.open(MCa_path) 
    MCb_file = uproot.open(MCb_path)
    MCb_data = MCb_file[tree_name]
#     MCb_data_DSID =  MCb_data["DSID"].array()
#     MCb_data_DSID = ak.to_numpy( MCb_data_DSID )
# #     if debug_removingEvents:
# #        MCb_data_DSID = np.delete(MCb_data_DSID, removeMCb_array, 0)
#     if isShapeNormReweight:
#         isSingleLepSample = np.equal(MCb_data_DSID, 410470) #410470 is a random number for the MCa vs MCa shape test.
#     else:
#         isSingleLepSample = np.equal(MCb_data_DSID, 410464) #410464(aMC); 410557(Herwig)
    #Getting a histogram called "h_Keep_SumAbsWeight_"
    # h_MCa_weight = MCa_file.keys()[3] 
    # h_MCb_SingleLep_weight = MCb_file.keys()[3] #fix: shoudn't ask people to hadd the single Lepton samples first and then the dilepton samples next...
    # # if isAddSample: h_MCb_diLep_weight = MCb_file.keys()[9]

    # h_MCa_weight = uproot.open(MCa_path)[h_MCa_weight]
    # h_MCb_SingleLep_weight = uproot.open(MCb_path)[h_MCb_SingleLep_weight]
    # if isAddSample: h_MCb_dileLep_weight = uproot.open(MCb_path)[h_MCb_diLep_weight]

    # Cross-section -> MOVE TO XSection_13TeV.txt file
    sigma_a = 831.75 * 0.543 #730pb, 0.54380
    ###TEST
    # sigma_a = 1 #730pb, 0.54380
    # sigma_b = (711*0.44037) + (712*0.10717)
    # sigma_a = 730 * 0.5438 
    #sigma_b = 831.76*0.43842 # 410464
    #sigma_b = 831.76*0.105 # 410465
    sigma_b = (831.76*0.105) + (831.76*0.43842) # 410464+410464

    #Get/Calculate the sum of weights
    # if weightFeature == "h_SumAbsWeight":
    #    # sum_a = h_MCa_weight.values()[0]
    #     sum_a = weightsa.sum() 
    #     sum_b_singleLep = 0.0
    #     sum_b_dilep = 0.0

    #     # for b in range(0, len(weightsb[:])):
    #     #     if isSingleLepSample[b]:
    #     #         sum_b_singleLep += weightsb[b]
    #     #     else:
    #     #         sum_b_dilep += weightsb[b]
    #     sum_b_singleLep = h_MCb_SingleLep_weight.values()[0]
    #     # if isAddSample: sum_b_dilep = h_MCb_dileLep_weight.values()[0]
#         print("Sum of Weight: MCa = {}, MCb(SigleLep) = {}, MCb(diLepton) = {}, weightsb.sum = {}".format(sum_a, sum_b_singleLep, sum_b_dilep, sum_b))
    if weightFeature == "EventWeight":
       # return the numpy arrays of the weights with all weights of 1.0
        sum_a = weightsa.sum()
        sum_b = weightsb.sum()
        # sum_b_singleLep = 0.0
        # sum_b_dilep = 0.0
        # for b in range(0, len(weightsb[:])):
        #    if isSingleLepSample[b]: 
        #       sum_b_singleLep += weightsb[b]
        #    else:   
        #       sum_b_dilep += weightsb[b]
#         print("Sum of Weight: MCa = {}, MCb single={}, MCb dil={}".format(sum_a, sum_b_singleLep, sum_b_dilep))
    else:
        sum_a = 1.0
        sum_b = 1.0
#         print("Sum of Weight: MCa = {}, MCb = {}".format(sum_a, sum_b))

    # Calculate effective luminosity
    # lumiscale_b_dilep = 0.0
    lumiscale_a = 1.0
    lumiscale_b = 1.0
    if weightFeature == "h_SumAbsWeight" or  weightFeature == "EventWeight":
        lumiscale_a = sigma_a/sum_a
        lumiscale_b = sigma_b/sum_b
        # lumiscale_b_singlelep = sigma_b/sum_b
#         if isShapeNormReweight: #for the closure checks (MCa vs MCa(shape+norm))
#             lumiscale_b_singlelep = np.array(ShapeReweight(MCa, features, 5), dtype=object)*(831.75 * 0.543)/sum_b_singleLep
#         else:
# #             lumiscale_b_singlelep = (831.76*0.43842)/sum_b_singleLep  
#             lumiscale_b_singlelep = (831.76*0.43842)/1  #711pb, genfelteff: 0.44037; #(831.76*0.43842)/sum_b for aMC generator #(730*0.43853) for Herwig
#         if isAddSample: lumiscale_b_dilep = (831.76*0.10717)/sum_b_dilep #712pb, genfelteff: 0.10717 #(831.76*0.10717)/sum_b for aMC generator #(730*0.10717) for Herwig
#         print("LumiScale:   lumiscale_a = {}".format(lumiscale_a))
#         print("LumiScale:   lumiscale_b_singlelep = {}".format(lumiscale_b_singlelep))
#         if isAddSample: print("LumiScale:   lumiscale_b_dilep = {}".format(lumiscale_b_dilep))

    # if weightFeature == "EventWeight":
    #     lumiscale_a = sigma_a/sum_a
    #     if isShapeNormReweight: #for the closure checks (MCa vs MCa(shape+norm))
    #         lumiscale_b_singlelep = np.array(ShapeReweight(MCa, features, 5), dtype=object)*(831.75 * 0.543)/sum_b
    #         print("lumiscale_b_singlelep with a shape reweight:{}".format(lumiscale_b_singlelep))
    #     else:
    #         lumiscale_b_singlelep = 730*0.43853/sum_b   #(831.76*0.43842)/sum_b for aMC generator
    #         print("lumiscale_b_singlelep:{}".format(lumiscale_b_singlelep))
    #     if isAddSample: 
    #         umiscale_b_dilep = 730*0.10547/sum_b #(831.76*0.10717)/sum_b for aMC generator
    #         print("umiscale_b_dilep:{}".format(umiscale_b_dilep))
    # elif weightFeature == "":
    #     lumiscale_a = np.ones( shape = ( len( weightsa ) ) )
    #     lumiscale_b = np.ones( shape = ( len( weightsb ) ) )
        # lumiscale_b_singlelep = 1.0
        # if isAddSample: umiscale_b_dilep = 1.0
        #lumiscale_b = sigma_b/sum_b
        #lumiscale_a = 1.0
        #lumiscale_b = 1.0
#         print("LumiScale:   lumiscale_a = {}".format(lumiscale_a))
#         print("LumiScale:   lumiscale_b = {}".format(lumiscale_b_singlelep))
    # lumiscale_b = sigma_b/sum_b

    # scale weights
    if weightFeature != "":
        weightsa[:] = weightsa[:] * lumiscale_a
        weightsb[:] = weightsb[:] * lumiscale_b
        # weightsa[:] = weightsa[:] * np.array(lumiscale_a, dtype=object)
        # weightsb[:] = weightsb[:] * np.array(lumiscale_b, dtype=object)
    # weightsb[:] = weightsb[:] * lumiscale_b_singlelep
        # weightsb[:] = weightsb[:] * lumiscale_b_singlelep
        # isSingleLepSample = np.equal(MCb_data['DSID'].array(), 410464)
    # if weightFeature == "h_SumAbsWeight" or weightFeature == "EventWeight":
    #     for b in range(0, len(weightsb[:])):
    #         if isShapeNormReweight:
    #             weightsb[b]= (weightsb[b] * lumiscale_b_dilep, weightsb[b] * lumiscale_b_singlelep[b])[isSingleLepSample[b]] 
    #         else:
    #             weightsb[b]= (weightsb[b] * lumiscale_b_dilep, weightsb[b] * lumiscale_b_singlelep)[isSingleLepSample[b]] 
            # weightsb[b]= (weightsb[b] * lumiscale_b_dilep, weightsb[b] * lumiscale_b_singlelep)[isSingleLepSample[b]]
    else:
        weightsa[:] = 1.0
        weightsb[:] = 1.0
        #weightsb[:] = weightsb[:] * lumiscale_b
    # else:
    #     weightsa[:] = weightsa[:] * lumiscale_a
    #     weightsb[:] = weightsb[:] * lumiscale_b

#     print("LumiScale:   weightsa = {}".format(weightsa))
#     print("LumiScale:   weightsb = {}".format(weightsb))
    outputs=[]
    outputs=[weightsa, weightsb]
    return outputs
    # return weightsa, weightsb

## Apply event cuts
# def EventsFilter(MCa, MCb):
#     boolArrayMCa = np.logical_and(MCa['m_VpT'].array() > 75, MCa['m_VpT'].array() < 200)

## Plot observables
def ObserablePlotter(MCa, MCb, weightsa, weightsb, features):

    # Construct the list of functions to generate event level compound observables
    #observables = [ mBB, dRBB, pTV, MET, pTB1, pTB2, dYWH, mJ ]
    #observables = { "mBB":mbb,     "dRBB":dRBB,   "pTV":pTV,     "MET":MET, 
    #                "PTb1":pTB1,   "pTB2":pTB2,   "dYWH":dYWH,   "mJ":mJ }
    observables = { "mBB":mbb }

    # Dictionary of histograms
    histDict = defautdict(TH1F)

    # Loop through events (rows - necessary as we need to compose event level compound observables)
    for row in len(MCa[:,0,0]):
        # Construct the TLorentzVectors
        TLV = FormulateLorentzVectors(row, MC, features)
        # Loop through functions
        for key,f in observables:
            histDict[key+"_MCa"].Fill(f(TLV)) #f(x) returns one scalar [y]

    for row in len(MCb[:,0,0]):
        # Construct the TLorentzVectors
        TLV = FormulateLorentzVectors(row, MC, features)
        # Loop through functions
        for key,f in observables:
            histDict[key+"_MCb"].Fill(f(TLV)) #f(x) returns one scalar [y]

    # Now loop over histograms and save to canvas
    for key,f in observables:

        can1 = TCanvas(key, key, 1200,800)
        
        # Save the Canvas
        cwd = os.getcwd()+"/"
        can1.SaveAs(cwd+feat+"_"+str(pdg)+".png")
        can1.SaveAs(cwd+feat+"_"+str(pdg)+".pdf")
        can1.SaveAs(cwd+feat+"_"+str(pdg)+".eps")
        can1.SaveAs(cwd+feat+"_"+str(pdg)+".root")

# Observables of key concern to VHbb
def mbb(TLV):

    # Loop through the TLV vector, get all pdgid == 5 
    # and then consider only the b-jets
    bjets = TLV[5]
    cjets = TLV[4]
    ljets = TLV[0]
    
    # If there are no b-jets return mbb = -1 and a weight of 1
    if len(bjets) + len(cjets) + len(ljets) < 2:
        return -1, 1.0

    # Sort into highest pT
    bjets_sorted = sorted(bjets, key = SortPtDescending)
    cjets_sorted = sorted(cjets, key = SortPtDescending)
    ljets_sorted = sorted(ljets, key = SortPtDescending)

    # Select based on jet availability
    if len(bjets_sorted) > 1:
        return (bjets_sorted[0] + bjets_sorted[1]).M(), 
    elif len(bjets_sorted) == 1 and len(cjets_sorted) > 0:
        return (bjets_sorted[0] +  cjets_sorted[0]).M()
    elif len(cjets_sorted) > 1:
        return (cjets_sorted[0] +  cjets_sorted[1]).M()
    elif len(cjets_sorted) == 1 and len(ljets_sorted) > 0:
        return (cjets_sorted[0] +  ljets_sorted[1]).M()
    else:
        return (ljets_sorted[0] +  ljets_sorted[1]).M()

# Sorting function
def SortPtDescending(lv):
    # vec = TLorentzVector()
    # vec.SetPtEtaPhiM(lv[0],lv[1],lv[2],lv[3])
    return lv.Pt()    

# Function to define lorentz vectors for each event
def FormulateLorentzVectors(MC, features, evt, pdgID):

    # MC is a row slice, so compose dictionary for each type of pdg ID
    TLV = []

    # Loop through the sliced data for said event
    #    -> Get the index of the pdgid feature
    pt_index = features.index('pt')
    eta_index = features.index('eta')
    phi_index = features.index('phi')
    mass_index = features.index('m')
    particle=-999
    if pdgID =='ele':
        particle=0
    if pdgID =='muon':
        particle=1
    # if pdgID =='tau':
    #     particle=2
    if pdgID =='b1':
        particle=2
    if pdgID =='b2':
        particle=3
    if pdgID =='3rdJet':
        particle=4
    if pdgID =='MET':
        particle=5
    # for num in range(len(pdgIDIndex_MC[0])):
    TLV.append( [MC[evt,particle,pt_index], 
                     MC[evt,particle,eta_index], 
                     MC[evt,particle,phi_index], 
                     MC[evt,particle,mass_index]])
    # pdgid_index = features.index('id')

    # boolPdigID = np.logical_and(MC[evt,:,pdgid_index] == pdgID, True)
    # pdgIDIndex_MC = np.where(boolPdigID == True)
    # for num in range(len(pdgIDIndex_MC[0])):
    #             TLV.append( [MC[evt,:,pt_index][np.array(pdgIDIndex_MC)[0][num]], 
    #                           MC[evt,:,eta_index][np.array(pdgIDIndex_MC)[0][num]], 
    #                           MC[evt,:,phi_index][np.array(pdgIDIndex_MC)[0][num]], 
    #                           MC[evt,:,mass_index][np.array(pdgIDIndex_MC)[0][num]]])
    # pdgids = MC[:,pdgid_index]
    # maxObj = len(pdgids)
    # for obj in range(0,maxObj):
    #     # Obtain pdgid
    #     pdgid = pdgids[obj]
    #     # Construct for each pdgid a TLorentzVector based on [pt, eta, phi, m]
    #     TLV[pdgid].append( TLorentzVector( MC[obj, features.index('TruthObj_pt')],
    #                                        MC[obj, features.index('TruthObj_phi')],
    #                                        MC[obj, features.index('TruthObj_eta')],
    #                                        MC[obj, features.index('TruthObj_m')] ) )
        
    return TLV

# # change event shapes for the clsure checks
def ShapeReweight(MC, features, pdgID):

    pdgid_index = features.index('TruthObj_pdgid')
    pt_index = features.index('TruthObj_pt')
    eta_index = features.index('TruthObj_eta')
    phi_index = features.index('TruthObj_phi')
    mass_index = features.index('TruthObj_m')

    pdgids = MC[:,:,pdgid_index]
    maxObj = len(pdgids)
    shapeWeight = []
    for evt in range(len(MC[:,:,pdgid_index])):
        TLVector = []
        boolPdgID_MC = np.logical_and(MC[evt,:,pdgid_index] == int(pdgID), True)
        pdgIdIndex_MC = np.where(boolPdgID_MC == True)
        # print("range:{}".format(range(np.array(pdgIdIndex_MC).shape[-1])))
        for num in range(len(pdgIdIndex_MC[0])):
            TLVector.append( [MC[evt,:,pt_index][np.array(pdgIdIndex_MC)[0][num]], 
                              MC[evt,:,eta_index][np.array(pdgIdIndex_MC)[0][num]], 
                              MC[evt,:,phi_index][np.array(pdgIdIndex_MC)[0][num]], 
                              MC[evt,:,mass_index][np.array(pdgIdIndex_MC)[0][num]]])
    # for num in range(np.array(pdgIdIndex_MC).shape[-1]):
    #     TLVector = []
    #     print("pid:{}".format(  MC[:,:,pdgid_index][np.array(pdgIdIndex_MC)[raw,column][0], np.array(pdgIdIndex_MC)[raw,column][1]] ))
        # TLVector.append( [MC[:,:,0][np.array(pdgIdIndex_MC)[...,num][0], np.array(pdgIdIndex_MC)[...,num][1]], 
        #                   MC[:,:,1][np.array(pdgIdIndex_MC)[...,num][0], np.array(pdgIdIndex_MC)[...,num][1]], 
        #                   MC[:,:,2][np.array(pdgIdIndex_MC)[...,num][0], np.array(pdgIdIndex_MC)[...,num][1]], 
        #                   MC[:,:,3][np.array(pdgIdIndex_MC)[...,num][0], np.array(pdgIdIndex_MC)[...,num][1]] ])
 
        TLVector = sorted(TLVector, key=itemgetter(0), reverse=True)
        if (len(TLVector)>1):
            bjet1 = TLorentzVector()
            bjet2 = TLorentzVector()
            bjet1.SetPtEtaPhiM(TLVector[0][0],TLVector[0][1],TLVector[0][2],TLVector[0][3])
            bjet2.SetPtEtaPhiM(TLVector[1][0],TLVector[1][1],TLVector[1][2],TLVector[1][3])
            shapeWeight.append(bjet1.DeltaR(bjet2))
        else:
            shapeWeight.append(1.0)

    return shapeWeight

def getObservable(MC, features):
    dictObs = dict()
    for index,feat in enumerate(features): #pt,eta,phi,m
        for particle in range(len(MC[1])):
            print("particle len:{}, {} th, feat:{}".format(len(MC[1]), particle, feat))

    pt_index = features.index('pt')
    mBB  = []
    pTV  = []
    dRbb = []
    dYWH = []
    for evt in range(len(MC[:,:,pt_index])):
        TLVector_bjets = []
        TLVector_cjets = []
        TLVector_Ele1 = []
        TLVector_Ele2 = []
        TLVector_Mu1 = []
        TLVector_Mu2 = []
        TLVector_MET = []
        wBoson = TLorentzVector()
        zBoson = TLorentzVector()
        hBoson = TLorentzVector()
        
        #store bjets
        TLVector_b1 = FormulateLorentzVectors(MC, features, evt, 'b1')
        #store bjets
        TLVector_b2 = FormulateLorentzVectors(MC, features, evt, 'b2')
        #store cjets
        #TLVector_Tau = FormulateLorentzVectors(MC, features, evt, 'tau')
        #store lepton 11
        TLVector_Ele =  FormulateLorentzVectors(MC, features, evt, 'ele')
        #store lepton 13
        TLVector_Mu = FormulateLorentzVectors(MC, features, evt, 'muon')
        #store MET 42
        TLVector_MET = FormulateLorentzVectors(MC, features, evt, 'MET')
        #store third jet
        TLVector_3rdJet = FormulateLorentzVectors(MC, features, evt, '3rdJet')
    
        TLVector_b1 = sorted(TLVector_b1, key=itemgetter(0), reverse=True)
        TLVector_b2 = sorted(TLVector_b2, key=itemgetter(0), reverse=True)
        #TLVector_Tau = sorted(TLVector_Tau, key=itemgetter(0), reverse=True)
        TLVector_Ele  = sorted(TLVector_Ele,  key=itemgetter(0), reverse=True)
        TLVector_Mu   = sorted(TLVector_Mu,  key=itemgetter(0), reverse=True)
        TLVector_MET   = sorted(TLVector_MET,  key=itemgetter(0), reverse=True)
        TLVector_3rdJet   = sorted(TLVector_3rdJet,  key=itemgetter(0), reverse=True)
        ##### reconstruct mBB ####
        if len(TLVector_b1)>0 and len(TLVector_b2)>0:
            bjet1 = TLorentzVector()
            bjet2 = TLorentzVector()
            bjet1.SetPtEtaPhiM(TLVector_b1[0][0],TLVector_b1[0][1],TLVector_b1[0][2],TLVector_b1[0][3])
            bjet2.SetPtEtaPhiM(TLVector_b2[0][0],TLVector_b2[0][1],TLVector_b2[0][2],TLVector_b2[0][3])
            mBB.append((bjet1+bjet2).M())
            dRbb.append(bjet1.DeltaR(bjet2))
            hBoson = bjet1+bjet2
        # elif len(TLVector_bjets)>0 and len(TLVector_cjets) >0:
        #     bjet1 = TLorentzVector()
        #     cjet1 = TLorentzVector()
        #     bjet1.SetPtEtaPhiM(TLVector_bjets[0][0],TLVector_bjets[0][1],TLVector_bjets[0][2],TLVector_bjets[0][3])
        #     cjet1.SetPtEtaPhiM(TLVector_cjets[0][0],TLVector_cjets[0][1],TLVector_cjets[0][2],TLVector_cjets[0][3])
        #     mBB.append((bjet1+cjet1).M())
        #     hBoson = bjet1+cjet1
        # elif len(TLVector_bjets)>0 and len(TLVector_ljets) >0:
        #     bjet1 = TLorentzVector()
        #     ljet1 = TLorentzVector()
        #     bjet1.SetPtEtaPhiM(TLVector_bjets[0][0],TLVector_bjets[0][1],TLVector_bjets[0][2],TLVector_bjets[0][3])
        #     ljet1.SetPtEtaPhiM(TLVector_ljets[0][0],TLVector_ljets[0][1],TLVector_ljets[0][2],TLVector_ljets[0][3])
        #     mBB.append((bjet1+ljet1).M())
        #     hBoson = bjet1+ljet1
        # elif len(TLVector_cjets)>1:
        #     cjet1 = TLorentzVector()
        #     cjet2 = TLorentzVector()
        #     cjet1.SetPtEtaPhiM(TLVector_cjets[0][0],TLVector_cjets[0][1],TLVector_cjets[0][2],TLVector_cjets[0][3])
        #     cjet2.SetPtEtaPhiM(TLVector_cjets[1][0],TLVector_cjets[1][1],TLVector_cjets[1][2],TLVector_cjets[1][3])
        #     mBB.append((cjet1+cjet2).M())
        #     hBoson = cjet1+cjet2
        # elif len(TLVector_cjets)>0 and len(TLVector_ljets) >0:
        #     cjet1 = TLorentzVector()
        #     ljet1 = TLorentzVector()
        #     cjet1.SetPtEtaPhiM(TLVector_cjets[0][0],TLVector_cjets[0][1],TLVector_cjets[0][2],TLVector_cjets[0][3])
        #     ljet1.SetPtEtaPhiM(TLVector_ljets[0][0],TLVector_ljets[0][1],TLVector_ljets[0][2],TLVector_ljets[0][3])
        #     mBB.append((cjet1+ljet1).M())
        #     hBoson = cjet1+ljet1
        # elif len(TLVector_ljets)>1:
        #     ljet1 = TLorentzVector()
        #     ljet2 = TLorentzVector()
        #     ljet1.SetPtEtaPhiM(TLVector_ljets[0][0],TLVector_ljets[0][1],TLVector_ljets[0][2],TLVector_ljets[0][3])
        #     ljet2.SetPtEtaPhiM(TLVector_ljets[1][0],TLVector_ljets[1][1],TLVector_ljets[1][2],TLVector_ljets[1][3])
        #     mBB.append((ljet1+ljet2).M())
        else:
            mBB.append(-999)
            dRbb.append(-999)
        #### reconstruct ptV ####
        if len(TLVector_MET) > 0 and len(TLVector_Ele) > 0:
            met = TLorentzVector()
            ele1 = TLorentzVector()
            met.SetPtEtaPhiM(TLVector_MET[0][0],0,TLVector_MET[0][2],0)
            ele1.SetPtEtaPhiM(TLVector_Ele[0][0],TLVector_Ele[0][1],TLVector_Ele[0][2],TLVector_Ele[0][3])
            pTV.append((met+ele1).Pt())
            wBoson =  met+ele1  

        elif len(TLVector_MET) > 0 and len(TLVector_Mu) > 0:
            met = TLorentzVector()
            mu1 = TLorentzVector()
            met.SetPtEtaPhiM(TLVector_MET[0][0],0,TLVector_MET[0][2],0)
            mu1.SetPtEtaPhiM(TLVector_Mu[0][0],TLVector_Mu[0][1],TLVector_Mu[0][2],TLVector_Mu[0][3])
            pTV.append((met+mu1).Pt())
            wBoson =  met+mu1
        '''
        elif len(TLVector_MET) > 0 and len(TLVector_Tau) > 0:
            met = TLorentzVector()
            tau = TLorentzVector()
            met.SetPtEtaPhiM(TLVector_MET[0][0],TLVector_MET[0][1],TLVector_MET[0][2],TLVector_MET[0][3])
            tau.SetPtEtaPhiM(TLVector_Tau[0][0],TLVector_Tau[0][1],TLVector_Tau[0][2],TLVector_Tau[0][3])
            pTV.append((met+tau).Pt())
            wBoson =  met+tau  
        '''
        # if len(TLVector_Ele1) > 0 and len(TLVector_Ele2) > 0:
        #     ele1 = TLorentzVector()
        #     ele2 = TLorentzVector()
        #     ele1.SetPtEtaPhiM(TLVector_Ele1[0][0],TLVector_Ele1[0][1],TLVector_Ele1[0][2],TLVector_Ele1[0][3])
        #     ele2.SetPtEtaPhiM(TLVector_Ele2[0][0],TLVector_Ele2[0][1],TLVector_Ele2[0][2],TLVector_Ele2[0][3])
        #     pTV.append((ele1+ele2).Pt())            
        # if len(TLVector_Mu1) > 0 and len(TLVector_Mu2) > 0:
        #     mu1 = TLorentzVector()
        #     mu2 = TLorentzVector()
        #     mu1.SetPtEtaPhiM(TLVector_Mu1[0][0],TLVector_Mu1[0][1],TLVector_Mu1[0][2],TLVector_Mu1[0][3])
        #     mu2.SetPtEtaPhiM(TLVector_Mu2[0][0],TLVector_Mu2[0][1],TLVector_Mu2[0][2],TLVector_Mu2[0][3])
        #     pTV.append((mu1+mu2).Pt()) 
        # else:
        #     pTV.append(-999)
        #### reconstruct dR(bb) ####
        # if len(TLVector_b1)>0 and len(TLVector_b2)>0:
        #     bjet1 = TLorentzVector()
        #     bjet2 = TLorentzVector()
        #     bjet1.SetPtEtaPhiM(TLVector_b1[0][0],TLVector_b1[0][1],TLVector_b1[0][2],TLVector_b1[0][3])
        #     bjet2.SetPtEtaPhiM(TLVector_b2[0][0],TLVector_b2[0][1],TLVector_b2[0][2],TLVector_b2[0][3])
        #     dRbb.append(bjet1.DeltaR(bjet2))
        # else:
        #     dRbb.append(-999)
        #### reconstruct dYWH ####
        dYWH.append(math.fabs(wBoson.Rapidity()- hBoson.Rapidity()))


    dictObs['mBB']  = mBB
    dictObs['pTV']  = pTV
    dictObs['dRbb'] = dRbb
    dictObs['dYWH'] = dYWH

    return dictObs


# Function for inducing a shape difference
def InduceShapeDiff(MC, MC_weights, features):

    # Extract pdg ID index number
    pdgid_index = features.index('TruthObj_pdgid')
    
    # Generate a flag masking array
    pdgid = MC[:,:,pdgid_index]
    print(pdgid)

    #pdgID filter arrays for MET with ID = 42
    #maskFilter = pdgid==57/115
    maskFilter = pdgid==42

    # Get a feature slice
    featSlice = MC[:, :, 0]

    # Apply filter
    featSlice = featSlice[maskFilter]
    
    # Flatten all objs with same pdgid into a 1D array for weighting with MC_weights array
    featSlice = featSlice.flatten()

    print("IndexuceShapeDiff::  {}".format(featSlice))
    print("IndexuceShapeDiff::  {}".format(MC_weights))


    # Multiply all weights by:  y =  MET*4000 + 0.8
    multFac  = featSlice * 4000
    print("IndexuceShapeDiff::  multFac = {}".format(multFac))
    multFac  = multFac + 0.8
    print("IndexuceShapeDiff::  multFac = {}".format(multFac))
    MC_weights = np.multiply(MC_weights, multFac)

    print("IndexuceShapeDiff::  {}".format(MC_weights))

    return MC_weights


