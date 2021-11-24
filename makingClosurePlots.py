from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import argparse
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import DSNNr_lib as DSNNr
from collections import defaultdict

# Global plot settings
from matplotlib import rc
import matplotlib.font_manager

from ROOT import TCanvas, TH1F, gROOT, gStyle, TLegend, TLorentzVector, TPad, TLine, gPad, TGraphErrors, TLatex
import ROOT as ROOT
import root_numpy as rn
import math 
from numpy import argmax
rc("font", family="serif")
rc("text", usetex=False)
rc("font", size=22)
rc("xtick", labelsize=15)
rc("ytick", labelsize=15)
rc("legend", fontsize=7.5)

def makingClosurePlots(args):
    if not os.path.split(os.getcwd())[-1] == args.global_name:
        if not os.path.isdir(args.global_name):
            os.makedirs(args.global_name)
        os.chdir(args.global_name)
    else:
        print("already in working directory")

    ROOT.PyConfig.IgnoreCommandLineOptions = True  # stop PyRoot hijacking -h WHY DOESNT THIS ALWAYS WORK
    ROOT.gROOT.SetBatch(True)  # Don't want to plot these to screen as we generate them
    ROOT.gROOT.SetStyle('ATLAS')
    # Remove stat box
    gStyle.SetOptStat(0)
    gStyle.SetTitleFontSize(0.08)
    gStyle.SetLabelSize(0.1)

    my_features = args.features.split(",")
    #print("my_features:{}".format(my_features))
    #n_features = len(args.features.split(","))
    #print(n_features)
    #model_scan = DSNNr.DS_model(n_features)
    #Loading model
    #model = keras.models.load_model("/hpcgpfs01/scratch/ftsai/DSNNr4gpu/"+args.global_name+"/saved_models/"+args.global_name+".ckpt")
    #model = keras.models.load_model("/hpcgpfs01/scratch/ftsai/DSNNr4gpu/"+args.global_name+"/saved_models_batchsize20000_2080/"+args.global_name+".ckpt")
    '''
    file_MCa = args.MCa.split('/')[-1]
    file_MCa = file_MCa.replace(".root", ".npz")
    file_MCb = args.MCb.split('/')[-1]
    file_MCb = file_MCb.replace(".root", ".npz")
    
    numpyMCa = np.load(file_MCa)
    numpyMCb = np.load(file_MCb)
    '''
    model = keras.models.load_model("/hpcgpfs01/scratch/ftsai/DSNNrBranch/Reco_WJet_MCade_5050/saved_models_batchsize20000_60nodes_5050/Reco_WJet_MCade_5050.ckpt")
    #model = keras.models.load_model("/hpcgpfs01/scratch/ftsai/DSNNrBranch/Reco_WJet_MCa_2080/saved_models_MCade_180Nodes_2080_20000_1000epochs/Reco_WJet_MCa_2080.ckpt")
    #get data
    npFile = np.load("/hpcgpfs01/scratch/ftsai/DSNNrBranch/Reco_WJet_MCade_5050/CombinedMCade5050.npz")
    #npFile = np.load("/hpcgpfs01/scratch/ftsai/DSNNrBranch/Reco_WJet_MCa_2080/CombinedMCade.npz")
    
    MCa = npFile["MCa"]
    MCb = npFile["MCb"]
    MCa_weights = npFile["MCa_weights"]
    MCb_weights = npFile["MCb_weights"]
    MCa_spectors = npFile["MCa_spectors"]
    MCb_spectors = npFile["MCb_spectors"]
    
    #numpyMCa = np.load("WJet_FullMET.hadd.npz")
    #numpyMCb = np.load("WJet_MGPyp_ResovedMET.hadd.npz")
    '''
    MCa = numpyMCa["MCa"]
    MCb = numpyMCb["MCb"]
    MCa_weights = numpyMCa["MCa_weights"]
    MCb_weights = numpyMCb["MCb_weights"]
    MCa_spectors = numpyMCa["MCa_spec"]
    MCb_spectors = numpyMCb["MCb_spec"]
    '''
    #ptV_MCa = MCa_spectors[:,0]
    #ptV_MCb = MCb_spectors[:,0]
    # (MCa, MCb, MCa_weights, MCb_weights, maxObjCount) = DSNNr.get_data(args)
    # print("MCa:{}, len:{}".format(MCa, len(MCa)))
    # (MCa, MCb, MCa_spec,MCb_spec, MCa_weights, MCb_weights, maxObjCount) = DSNNr.get_data(args)
    #handle data
    #X_train, X_test, Y_train, Y_test, train_weights, test_weights, S_train, S_test = DSNNr.handle_data(args, MCa, MCb, MCa_weights, MCb_weights, MCa_spectors, MCb_spectors)
    
    splitMC = np.load("/hpcgpfs01/scratch/ftsai/DSNNrBranch/Reco_WJet_MCade_5050/CombinedSplitMCade5050.npz")
    #splitMC = np.load("/hpcgpfs01/scratch/ftsai/DSNNrBranch/Reco_WJet_MCa_2080/CombinedSplitMCade.npz")
    X_train = splitMC["X_train"]
    X_test = splitMC["X_test"]
    Y_train = splitMC["Y_train"]
    Y_test = splitMC["Y_test"]
    train_weights = splitMC["W_train"]
    test_weights = splitMC["W_test"]
    S_train = splitMC["S_train"]
    S_test = splitMC["S_test"]
    

    Y_train = argmax(Y_train, axis=1)
    Y_test = argmax(Y_test, axis=1)
    X_train_MCa = X_train[Y_train == 0]
    X_test_MCa = X_test[Y_test == 0]
    train_weights_MCa = train_weights[Y_train == 0]
    test_weights_MCa = test_weights[Y_test == 0]
    #get predictions on testing + training data
    preds = model.predict(MCa)
    predsB = model.predict(MCb)
    #print("FY preds:{}".format(preds)) 
    NNweights = preds[:, 1] / preds[:, 0]
    Pa_MCa = preds[:, 0].flatten()
    Pb_MCa = preds[:, 1].flatten()
    Pb_MCb = predsB[:, 0].flatten()
    Pa_MCb = predsB[:, 1].flatten()
    print("Probability Pa:{}, Pb:{}".format(Pa_MCa, Pb_MCb))
    print("NNweights:{}, preds[:, 1]:{}, preds[:, 0]:{}".format(NNweights, preds[:, 1], preds[:, 0]))
    flag_testNtraining = True
    flag_residualError = False

    if flag_testNtraining:
        #get predictions on testing data
        preds_test = model.predict(X_test_MCa) 
        Pa_MCa_test = preds_test[:, 0].flatten()
        NNweights_test = preds_test[:, 1] / preds_test[:, 0]
        print("NNweights_test:{}, preds_test[:, 1]:{}, preds_test[:, 0]:{}".format(NNweights_test, preds_test[:, 1], preds_test[:, 0]))
        #get predictions on training data
        preds_train = model.predict(X_train_MCa) 
        Pa_MCa_train = preds_train[:, 0].flatten()
        NNweights_train = preds_train[:, 1] / preds_train[:, 0]
        print("NNweights_train:{}, preds_test[:, 1]:{}, preds_test[:, 0]:{}, len:{}".format(NNweights_train, preds_train[:, 1], preds_train[:, 0], len(NNweights_train)))
        print("NNweight_train shape:{}, test shape:{}".format(np.array(NNweights_train).shape, np.array(NNweights_test).shape))
    #get MCa and MCb 
    # preds_MCa = model.predict(MCa)
    # MCaWeight = preds_MCa[:,0]
    # preds_MCb = model.predict(MCb) 
    # MCbWeight = preds_MCb[:,0]
    # print("preds_MCa[:,0]:{}, preds_MCb[:,0]:{}, Prob of MCa:{}".format(preds_MCa[:,0], preds_MCb[:,0], preds[:, 0]))
    
    # pdgids = [0]
    # pdgids = [-11,11,-13,13,-15,15,0,4,5,42,100,200,204,205]
    # pdgid_index = my_features.index('TruthObj_pdgid')

    # pdgid_r = MCa[:,:,pdgid_index]
    # pdgid_test =  X_test_MCa[:,:,pdgid_index]
    # pdgid_train = X_train_MCa[:,:,pdgid_index]
    # pdgid_a = MCa[:,:,pdgid_index]
    # pdgid_b = MCb[:,:,pdgid_index]

    for index,feat in enumerate(my_features):
        for particle in range(len(MCa[1])):
            print("particle len:{}, {} th, feat:{}".format(len(MCa[1]), particle, feat))
            # Extract the feature slices
            featSlice_a = MCa[:, particle, index]
            featSlice_b = MCb[:, particle, index]
            featSlice_r = MCa[:, particle, index]


            if flag_testNtraining:
                featSlice_test = X_test_MCa[:, particle, index]
                featSlice_train = X_train_MCa[:, particle, index]

            #filter negative values
            filter_a = MCa[:, particle, index] > -5
            filter_b = MCb[:, particle, index] > -5
            filter_r = MCa[:, particle, index] > -5
            if flag_testNtraining:
                filter_test =  X_test_MCa[:,particle, index] > -5
                filter_train = X_train_MCa[:,particle,index] > -5
            print("filter_a:{}".format(filter_a))
            #pdgID filter arrays
            # filter_a = pdgid_a==pdg
            # filter_b = pdgid_b==pdg
            # filter_r = pdgid_r==pdg
            # filter_test =  pdgid_test==pdg
            # filter_train = pdgid_train==pdg
            # filter_a = np.logical_and(pdgid_a==pdg, filterDummy_a)
            # filter_b = np.logical_and(pdgid_b==pdg, filterDummy_b)
            # filter_r = np.logical_and(pdgid_r==pdg, filterDummy_r)
            # Request only values > 0
            featSlice_a_cut = featSlice_a[filter_a]
            featSlice_b_cut = featSlice_b[filter_b]
            featSlice_r_cut = featSlice_r[filter_r]

            if flag_testNtraining:
                featSlice_test_cut = featSlice_test[filter_test]
                featSlice_train_cut = featSlice_train[filter_train]

            # Flatten all objs with same pdgid into a 1D array
            featSlice_a_flat = featSlice_a_cut.flatten()
            featSlice_b_flat = featSlice_b_cut.flatten()
            featSlice_r_flat = featSlice_r_cut.flatten()

            # featSlice_a_flat = featSlice_a.flatten()
            # featSlice_b_flat = featSlice_b.flatten()
            # featSlice_r_flat = featSlice_r.flatten()
            print("MCa, obj:{}, feat:{}, array:{}".format(particle, feat, featSlice_a))
            print("flatten MCa, obj:{}, feat:{}, array:{}".format(particle, feat, featSlice_a_flat))

            if flag_testNtraining:
                featSlice_test_flat = featSlice_test_cut.flatten()
                featSlice_train_flat = featSlice_train_cut.flatten()
            if len(featSlice_a_flat) == 0:
                continue
            minElem = np.amin(featSlice_b_flat)
            maxElem = np.amax(featSlice_b_flat)
            nDiv = 50
            print("minElem:{}, maxElem:{}".format(minElem, maxElem))
            # Create the weight array of the same shape as the featSlice_... before flattening
            weightsa_expand = MCa_weights[filter_a]
            weightsb_expand = MCb_weights[filter_b]
            weightsw_expand = NNweights[filter_a] 
            # weightsa_expand = np.repeat( np.expand_dims(MCa_weights, axis=1), repeats=maxObjCount, axis = 1  )
            # weightsb_expand = np.repeat( np.expand_dims(MCb_weights, axis=1), repeats=maxObjCount, axis = 1  )
            # weightsw_expand = np.repeat( np.expand_dims(NNweights, axis=1), repeats=maxObjCount, axis = 1  )
            print("MCa_weight:{}, NNWeights:{}".format(np.array(MCa_weights).shape, np.array(NNweights).shape) )
            print("featSlice_a_flat shape:{}, weightsa_expand:{}, NN expand:{}".format(np.array(featSlice_a_flat).shape, np.array(weightsa_expand).shape, np.array(weightsw_expand).shape))
            '''
            weightsMCa_expand = np.repeat( np.expand_dims(MCaWeight, axis=1), repeats=maxObjCount, axis = 1  )
            weightsMCb_expand = np.repeat( np.expand_dims(MCbWeight, axis=1), repeats=maxObjCount, axis = 1  )
            '''
            weightsr_expand = weightsw_expand * weightsa_expand
            '''
            MCaWeight_expand = weightsMCa_expand * weightsa_expand
            MCbWeight_expand = weightsMCb_expand * weightsb_expand
            '''

            if flag_testNtraining:
                # train_weights_exp = np.repeat( np.expand_dims(train_weights_MCa, axis=1), repeats=maxObjCount, axis = 1  )
                # test_weights_exp = np.repeat( np.expand_dims(test_weights_MCa, axis=1), repeats=maxObjCount, axis = 1  )
                # weightTest_expand = np.repeat( np.expand_dims(NNweights_test, axis=1), repeats=maxObjCount, axis = 1  )
                # weighttTrains_expand = np.repeat( np.expand_dims(NNweights_train, axis=1), repeats=maxObjCount, axis = 1  )
                train_weights_exp = train_weights_MCa[filter_train]
                test_weights_exp =test_weights_MCa[filter_test]
                weightTest_expand = NNweights_test[filter_test]
                weighttTrains_expand = NNweights_train[filter_train]
                weightTestR_expand = weightTest_expand* test_weights_exp
                weightTrainR_expand = weighttTrains_expand*train_weights_exp
                print("train_weights_exp shape:{}, test_weights_exp:{}".format(np.array(train_weights_exp).shape, np.array(test_weights_exp).shape)) 
                print("weighttTrains_expand shape:{}, weightTest_expand:{}".format(np.array(weighttTrains_expand).shape, np.array(weightTest_expand).shape))
                print("weightTrainR_expand:{}, weightTestR_expand:{}".format(np.array(weightTrainR_expand).shape, np.array(weightTestR_expand).shape)) #weightTrainR_expand:(531361,), weightTestR_expand:(133215,)
                print("histo flatten train:{}, test:{}".format(np.array(featSlice_train_flat).shape, np.array(featSlice_test_flat).shape)) #flatten test:(666075,), train:(2656805,)
            # Create a canvas
            # can1 = TCanvas(feat+"_"+str(pdg), feat+"_"+str(pdg), 400,600)
            can1 = TCanvas(feat+"_"+str(particle), feat+"_"+str(particle))
            # can1.SetCanvasSize(1500, 1500)
            # can1.SetWindowSize(500, 500)
            mainPad =  TPad("mainPad", "top",    0.0, 0.37, 1.0, 1.00)
            ratioPad = TPad("ratioPad","bottom", 0.0, 0.02, 1.0, 0.37)

            mainPad.SetBottomMargin(0.01) 
            ratioPad.SetTopMargin(0.08) 
            ratioPad.SetBottomMargin(0.15)

            #if feat == 'pt' or feat == 'm' or feat == 'px' or feat == 'py' or feat == 'pz' or feat == 'E':   
                #mainPad.SetLogy()
            mainPad.SetLogy()
            mainPad.Draw()
            mainPad.cd()
            # Create histograms
            hista = TH1F(feat+"_a_"+str(particle), feat+"_a_"+str(particle), nDiv, minElem, maxElem)
            histb = TH1F(feat+"_b_"+str(particle), feat+"_b_"+str(particle), nDiv, minElem, maxElem)

            histr = TH1F(feat+"_r_"+str(particle), feat+"_r_"+str(particle), nDiv, minElem, maxElem)
            #histw = TH1F(feat+"_w_"+str(particle), feat+"_r_"+str(particle), nDiv, minElem, maxElem)
            histTest = TH1F(feat+"_test_"+str(particle), feat+"_test_"+str(particle), nDiv, minElem, maxElem)
            histTrain = TH1F(feat+"_train_"+str(particle), feat+"_train_"+str(particle), nDiv, minElem, maxElem)
            '''
            histMCa = TH1F(feat+"_MCa", feat+"_MCa", nDiv, minElem, maxElem)
            histMCb = TH1F(feat+"_MCb", feat+"_MCb", nDiv, minElem, maxElem)
            b
            '''
            minWeight = np.amin(weightsw_expand)
            maxWeight = np.amax(weightsw_expand)
            minPro = np.amin(Pa_MCa)
            maxPro = np.amax(Pb_MCa)
            minPro2 = np.amin(Pa_MCa_test)
            maxPro2 = np.amax(Pa_MCa_test) 
            print('min:{},max:{}'.format(minWeight, maxWeight))
            histw = TH1F(feat+"_w"+str(particle), feat+"_w"+str(particle), nDiv, minWeight, maxWeight)
            histt = TH1F(feat+"_test"+str(particle), feat+"_test"+str(particle), nDiv, minWeight, maxWeight)
            histPa = TH1F(feat+"_histPa"+str(particle), feat+"_histPa"+str(particle), nDiv, minPro, maxPro)
            histPa_test = TH1F(feat+"_histPa_test"+str(particle), feat+"_histPa_test"+str(particle), nDiv, minPro2, maxPro2)
            histPa_train = TH1F(feat+"_histPa_train"+str(particle), feat+"_histPa_train"+str(particle), nDiv, minPro2, maxPro2)
            histPb = TH1F(feat+"_histPb"+str(particle), feat+"_histPb"+str(particle), nDiv, minPro, maxPro)
            rn.fill_hist(histw, weightsw_expand, np.ones( shape = (len(weightsw_expand))))
            rn.fill_hist(histt, weightTest_expand, np.ones( shape = (len(weightTest_expand))))
            rn.fill_hist(histPa, Pa_MCa , np.ones( shape = (len(Pa_MCa))))
            rn.fill_hist(histPa_test, Pa_MCa_test, np.ones( shape = (len(Pa_MCa_test))))
            rn.fill_hist(histPa_train, Pa_MCa_train, np.ones( shape = (len(Pa_MCa_train))))
            rn.fill_hist(histPb, Pb_MCa, np.ones( shape = (len(Pb_MCa))))
            rn.fill_hist(hista, featSlice_a_flat, weightsa_expand)
            rn.fill_hist(histb, featSlice_b_flat, weightsb_expand) 
            rn.fill_hist(histr, featSlice_r_flat, weightsr_expand)
            histPa.Scale(1/histPa.Integral())
            histPb.Scale(1/histPb.Integral())
            '''
            rn.fill_hist(histMCa, featSlice_a_flat, MCaWeight_expand[filter_a])
            rn.fill_hist(histMCb, featSlice_b_flat, MCbWeight_expand[filter_b])
            rn.fill_hist(histw, weightsw_expand[filter_r], np.ones( shape = (len(weightsw_expand[filter_r]))))
            '''
            if flag_testNtraining:
                rn.fill_hist(histTest, featSlice_test_flat, weightTestR_expand)
                rn.fill_hist(histTrain, featSlice_train_flat, weightTrainR_expand)
                histTest.Scale(1/histTest.Integral())
                histTrain.Scale(1/histTrain.Integral())
                histb.Scale(1/histb.Integral())
                hista.Scale(1/hista.Integral())
            # histr.DrawCopy("hist")
            # Draw histograms
            if flag_testNtraining:
                #Testing
                histTest.GetYaxis().SetTitle("#frac{d#sigma}{d"+feat+"}")
                histTest.SetFillColorAlpha(ROOT.kBlue, 0.25)
                histTest.SetLineColor(ROOT.kBlue)
                histTest.GetYaxis().SetTitleSize(0.04)
                histTest.GetYaxis().SetLabelSize(0.02)
                histTest.GetXaxis().SetLabelSize(0.02)
                histTest.DrawCopy("hist")

                hista.GetYaxis().SetTitle("#frac{d #sigma}{d"+feat+"}")
                # hista.GetXaxis().SetTitle(feat)
                # hista.SetFillStyle(1001)
                # hista.SetFillStyle(3004)
                # hista.SetFillColor(ROOT.kAzure+7)
                hista.SetFillColorAlpha(ROOT.kAzure+7, 0.25)
                hista.SetLineColor(ROOT.kAzure+7)
                hista.GetYaxis().SetTitleSize(0.04)
                hista.GetYaxis().SetLabelSize(0.02)
                hista.GetXaxis().SetLabelSize(0.02)
                hista.GetYaxis().SetTitleOffset(0.7)
                hista.DrawCopy("hist SAME")
    
                #Training
                histTrain.GetYaxis().SetTitle("#frac{d#sigma}{d"+feat+"}")
                histTrain.SetFillColorAlpha(ROOT.kMagenta-2, 0.25)
                histTrain.SetLineColor(ROOT.kMagenta-2)
                histTrain.GetYaxis().SetTitleSize(0.04)
                histTrain.GetYaxis().SetLabelSize(0.02)
                histTrain.GetXaxis().SetLabelSize(0.02)
                histTrain.DrawCopy("hist SAME")

                histb.GetYaxis().SetTitle("#frac{d#sigma}{d"+feat+"}")
                histb.SetFillColorAlpha(ROOT.kOrange-3, 0.25)
                histb.SetLineColor(ROOT.kOrange-3)
                histb.GetYaxis().SetTitleSize(0.04)
                histb.GetYaxis().SetLabelSize(0.02)
                histb.GetXaxis().SetLabelSize(0.02)
                histb.DrawCopy("hist SAME")
            else:
            #    MCa
                hista.GetYaxis().SetTitle("#frac{d #sigma}{d"+feat+"}")
                # hista.GetXaxis().SetTitle(feat)
                # hista.SetFillStyle(1001)
                # hista.SetFillStyle(3004)
                # hista.SetFillColor(ROOT.kAzure+7)
                hista.SetFillColorAlpha(ROOT.kAzure+7, 0.25)
                hista.SetLineColor(ROOT.kAzure+7)
                hista.GetYaxis().SetTitleSize(0.04)
                hista.GetYaxis().SetLabelSize(0.02)
                hista.GetXaxis().SetLabelSize(0.02)
                hista.GetYaxis().SetTitleOffset(0.7)
                hista.DrawCopy("hist")
        
                #    MCb
                histb.GetYaxis().SetTitle("#frac{d#sigma}{d"+feat+"}")
                # histb.GetXaxis().SetTitle(feat)
                # histb.SetFillStyle(1001)
                histb.SetFillColorAlpha(ROOT.kOrange-3, 0.25)
                # histb.SetFillColor(ROOT.kOrange-3)
                histb.SetLineColor(ROOT.kOrange-3)
                # histb.SetFillStyle(3004)
                histb.GetYaxis().SetTitleSize(0.04)
                histb.GetYaxis().SetLabelSize(0.02)
                histb.GetXaxis().SetLabelSize(0.02)
                # histb.GetYaxis().SetTitleOffset(0.65)
                histb.DrawCopy("hist SAME")

                histr.GetYaxis().SetTitle("#frac{d #sigma}{d"+feat+"}")
                histr.SetLineColor(2)
                histr.DrawCopy("hist SAME")
                histr.GetYaxis().SetTitle("#frac{d #sigma}{d"+feat+"}")
                histr.SetLineColor(2)
                histr.DrawCopy("hist SAME")
            if flag_testNtraining:
                tt1 = ROOT.TLatex()
                tt1.SetNDC()
                tt1.SetTextSize(0.04)
                tt1.DrawLatex(0.45 ,0.75 ,"p-value(test, MCb):"+str(format(histTest.Chi2Test(histb,"WW P"),'.1E')))
                tt2 = ROOT.TLatex()
                tt2.SetNDC()
                tt2.SetTextSize(0.04)
                tt2.DrawLatex(0.45 ,0.70 ,"p-value(train, MCb):"+str(format(histTrain.Chi2Test(histb,"WW P"),'.1E')))
                tt3 = ROOT.TLatex()
                tt3.SetNDC()
                tt3.SetTextSize(0.04)
                tt3.DrawLatex(0.45 ,0.65 ,"p-value(MCa, MCb):"+str(format(hista.Chi2Test(histb,"WW P"),'.1E')))
            else:
                tt1 = ROOT.TLatex()
                tt1.SetNDC()
                tt1.SetTextSize(0.04)
                tt1.DrawLatex(0.45 ,0.75 ,"p-value(MCa*NN, MCb):"+str(format(histr.Chi2Test(histb,"WW P"),'.1E')))
            n_ex=[]
            n_ey=[]
            n_x=[]
            n_y=[]
            n_y2=[]
            residualDiff = []
            residualDiff_Error = []
            SquareResidueDiff = []
            ErrorBand1 = []
            ErrorBand3 = []
            ErrorBand5 = []
            for hisBin in range(histb.GetNbinsX()):
                # print("hisBin:{}, value:{}".format(hisBin, histb.GetBinLowEdge(hisBin))) 
                residualDiff.append(histb.GetBinContent(hisBin) - histr.GetBinContent(hisBin))
                SquareResidueDiff.append(pow((histb.GetBinContent(hisBin) - histr.GetBinContent(hisBin)), 2))
                n_x.append(histb.GetBinLowEdge(hisBin)+ 0.5*(histb.GetBinLowEdge(hisBin+1)-(histb.GetBinLowEdge(hisBin))))
                n_y.append(1) #center at 1 in the y-axis
                n_y2.append(0) 
                n_ex.append((histb.GetBinLowEdge(hisBin+1)-histb.GetBinLowEdge(hisBin))*0.5)
            
                if histb.GetBinContent(hisBin)!=0:
                    n_ey.append(math.sqrt(pow(histb.GetBinError(hisBin),2)+pow(histTest.GetBinError(hisBin),2))/(histb.GetBinContent(hisBin)+histTest.GetBinContent(hisBin)))
                else:
                    n_ey.append(2)
                if math.sqrt(pow(histb.GetBinError(hisBin),2)+pow(histr.GetBinError(hisBin),2))!=0:
                    residualDiff_Error.append((histb.GetBinContent(hisBin) - histr.GetBinContent(hisBin))/math.sqrt(pow(histb.GetBinError(hisBin),2)+pow(histr.GetBinError(hisBin),2)))
                else:
                    residualDiff_Error.append(0)
                # if pow(histb.GetBinError(hisBin),2)+pow(histr.GetBinError(hisBin),2)!=0:
                #     ErrorBand1.append((histb.GetBinContent(hisBin) - histr.GetBinContent(hisBin))/(math.sqrt(pow(histb.GetBinError(hisBin),2)+pow(histr.GetBinError(hisBin),2))/(histb.GetBinContent(hisBin)+histr.GetBinContent(hisBin))))
                # else:
                #     ErrorBand1.append(0)

            gr = TGraphErrors(50,np.array(n_x, dtype=float),np.array(n_y, dtype=float),np.array(n_ex, dtype=float),np.array(n_ey, dtype=float))
            gr.GetXaxis().SetLimits(minElem,maxElem)
            gr.GetYaxis().CenterTitle(1)
            gr.GetYaxis().SetTitleOffset(0.3)
            gr.GetYaxis().SetTitleSize(0.1)
            gr.GetXaxis().SetTitle(feat)
            gr.GetXaxis().SetTitleOffset(0.65)
            gr.GetXaxis().SetTitleSize(0.15)
            gr.GetYaxis().SetLabelSize(0.04)
            gr.SetMarkerStyle(1)
            gr.SetFillColorAlpha(ROOT.kPink+6, 0.25)
            gr.SetLineColor(ROOT.kPink+6)

            if flag_testNtraining:
                gr.GetYaxis().SetTitle("w.r.t MC_{b}")
                gr.GetYaxis().SetRangeUser(0,2)
            else:
                gr.GetYaxis().SetTitle("w.r.t MC_{b}")
                gr.GetYaxis().SetRangeUser(0.8,1.2)


            # gr2 = TGraphErrors(50, )
            # Now define the TLegend
            # legend = TLegend(0.2,0.65,0.4,0.85)
            legend = TLegend(0.68,0.65,0.95,0.85)
            legend.SetBorderSize(0)
            #legend.SetTextSize(0)
            if flag_testNtraining: 
               legend.AddEntry(histb, "MC_{b}", "l")
               legend.AddEntry(hista, "MC_{a}", "l")
               legend.AddEntry(histTest, "MCa(testing) * NN", "l")
               legend.AddEntry(histTrain, "MCa(training) * NN", "l")

            else:
                legend.AddEntry(hista, "MC_{a}", "f")
                legend.AddEntry(histb, "MC_{b}", "f")
                legend.AddEntry(histr,"MC_{a} * NN", "l")
            legend.AddEntry(gr, "(Test+MCb)Stat Unc.")
            # legend.AddEntry(histb, "MC_{b}", "f")
            legend.SetTextSize(0.045)        
            legend.Draw()
            can1.cd()
            ratioPad.Draw()
            ratioPad.cd()
            gr.Draw("a2")
            ratio_baseline = hista.Clone("ratio_hista_"+feat)
            ratio_baseline.Divide(histb)
            ratio_baseline.SetFillColorAlpha(0, 0.25)
            ratio_baseline.SetMarkerStyle(86)
            ratio_baseline.SetMarkerColor(ROOT.kAzure+7)
            ratio_baseline.SetMarkerSize(0.7)
            ratio_baseline2 = histr.Clone("ratio_histr_"+feat)
            ratio_baseline2.Divide(histb)
            ratio_baseline2.SetFillColorAlpha(0, 0.25)
            ratio_baseline2.SetMarkerStyle(86)
            ratio_baseline2.SetMarkerColor(2)
            ratio_baseline2.SetMarkerSize(0.7)
            if flag_testNtraining:
                ratio_baseline3 = histTest.Clone("ratio_histTest_"+feat)
                ratio_baseline3.Divide(histb)
                # ratio_baseline3.SetFillColorAlpha(0, 0.25)
                # ratio_baseline3.SetMarkerStyle(86)
                ratio_baseline3.SetMarkerColor(ROOT.kBlue)
                ratio_baseline3.SetMarkerSize(0.7)
                ratio_baseline4 = histTrain.Clone("ratio_histTrain_"+feat)
                ratio_baseline4.Divide(histb)
                # ratio_baseline4.SetFillColorAlpha(0, 0.25)
                # ratio_baseline4.SetMarkerStyle(86)
                ratio_baseline4.SetMarkerColor(ROOT.kMagenta-2)
                ratio_baseline4.SetMarkerSize(0.7)
            ratio_baseline5 = histb.Clone("ratio_histr_"+feat)
            ratio_baseline5.Divide(histb)
            # master_data.GetBinContent(hisBin)*h_2.GetBinContent(hisBin)

            #TLine
            line = TLine(ratio_baseline.GetXaxis().GetXmin(),1,ratio_baseline.GetXaxis().GetXmax(),1)
            line.SetLineWidth(2)
            line.SetLineStyle(2)
            line.Draw("SAME")
            # ratio_baseline.SetLineWidth(1)
            # ratio_baseline.SetLineColor(14)
            
            if flag_testNtraining:
                ratio_baseline3.SetTitle("")
                ratio_baseline3.GetYaxis().SetTitle("w.r.t MC_{b}")
                ratio_baseline3.GetYaxis().SetRangeUser(0,2)
                ratio_baseline3.GetYaxis().CenterTitle(1)
                ratio_baseline3.GetYaxis().SetTitleOffset(0.5)
                ratio_baseline3.GetYaxis().SetTitleSize(0.1)
                ratio_baseline3.GetXaxis().SetTitle(feat)
                ratio_baseline3.GetXaxis().SetTitleOffset(0.65)
                ratio_baseline3.GetXaxis().SetTitleSize(0.15)
                ratio_baseline3.GetYaxis().SetLabelSize(0.04)
                ratio_baseline3.Draw("hist p SAME")
                ratio_baseline4.SetTitle("")
                ratio_baseline4.GetYaxis().SetTitle("w.r.t MC_{b}")
                ratio_baseline4.GetYaxis().SetRangeUser(0, 2)
                ratio_baseline4.GetYaxis().CenterTitle(1)
                ratio_baseline4.GetYaxis().SetTitleOffset(0.5)
                ratio_baseline4.GetYaxis().SetTitleSize(0.1)
                ratio_baseline4.GetXaxis().SetTitle(feat)
                ratio_baseline4.GetXaxis().SetTitleOffset(0.65)
                ratio_baseline4.GetXaxis().SetTitleSize(0.15)
                ratio_baseline4.GetYaxis().SetLabelSize(0.04)
                ratio_baseline4.Draw("hist p SAME")
                ratio_baseline.SetTitle("")
                ratio_baseline.GetYaxis().SetTitle("w.r.t MC_{b}")
                ratio_baseline.GetYaxis().SetRangeUser(0, 2)
                ratio_baseline.GetYaxis().CenterTitle(1)
                ratio_baseline.GetYaxis().SetTitleOffset(0.5)
                ratio_baseline.GetYaxis().SetTitleSize(0.1)
                ratio_baseline.GetXaxis().SetTitle(feat)
                ratio_baseline.GetXaxis().SetTitleOffset(0.65)
                ratio_baseline.GetXaxis().SetTitleSize(0.15)
                ratio_baseline.GetYaxis().SetLabelSize(0.04)
                ratio_baseline.Draw("hist p SAME")

            else:
                ratio_baseline.SetTitle("")
                ratio_baseline.GetYaxis().SetTitle("w.r.t MCb")
                ratio_baseline.GetYaxis().SetRangeUser(0.8,1.2)
                ratio_baseline.GetYaxis().CenterTitle(1)
                ratio_baseline.GetYaxis().SetTitleOffset(0.5)
                ratio_baseline.GetYaxis().SetTitleSize(0.1)
                ratio_baseline.GetXaxis().SetTitle(feat)
                ratio_baseline.GetXaxis().SetTitleOffset(0.65)
                ratio_baseline.GetXaxis().SetTitleSize(0.15)
                ratio_baseline.GetYaxis().SetLabelSize(0.04)
        
                ratio_baseline2.SetTitle("")
                ratio_baseline2.GetYaxis().SetTitle("w.r.t MCb")
                ratio_baseline2.GetYaxis().SetRangeUser(0.8,1.2)
                ratio_baseline2.GetYaxis().CenterTitle(1)
                ratio_baseline2.GetYaxis().SetTitleOffset(0.5)
                ratio_baseline2.GetYaxis().SetTitleSize(0.1)
                ratio_baseline2.GetXaxis().SetTitle(feat)
                ratio_baseline2.GetXaxis().SetTitleOffset(0.65)
                ratio_baseline2.GetXaxis().SetTitleSize(0.15)
                ratio_baseline2.GetYaxis().SetLabelSize(0.04)
                ratio_baseline2.Draw("hist p SAME")
                ratio_baseline.Draw("hist p SAME")
            # line.Draw("SAME")
            # can1.cd().SetLogy()
            can1.cd()
            can1.Draw()
            can1.Update()
            cwd = os.getcwd()+"/"
            if flag_testNtraining:
                can1.SaveAs(cwd+"testMCade180_"+str(particle)+"_"+feat+".pdf")
                can1.SaveAs(cwd+"testMCade180_"+str(particle)+"_"+feat+".png")
            else:
                can1.SaveAs(cwd+"test2_"+str(particle)+"_"+feat+".pdf")
                can1.SaveAs(cwd+"test2_"+str(particle)+"_"+feat+".png")
            can1.Close()
            can2 = TCanvas(feat+"c2_"+str(particle), feat+"_"+str(particle))
            histw.GetXaxis().SetTitle("NN Weights")
            histw.SetLineColor(ROOT.kOrange-3)
            histw.GetYaxis().SetTitleSize(0.04)
            histw.GetYaxis().SetLabelSize(0.02)
            histw.GetXaxis().SetLabelSize(0.02)
            histw.DrawCopy("hist")
            can2.SaveAs(cwd+"Weight180_"+str(particle)+"_"+feat+".png")
            can2.Close()
            can3 = TCanvas(feat+"c3_"+str(particle), feat+"_"+str(particle))
            histt.GetXaxis().SetTitle("NN Weights")
            histt.SetLineColor(ROOT.kOrange-3)
            histt.GetYaxis().SetTitleSize(0.04)
            histt.GetYaxis().SetLabelSize(0.02)
            histt.GetXaxis().SetLabelSize(0.02)
            histt.DrawCopy("hist")
            can3.SaveAs(cwd+"WeightTest180_"+str(particle)+"_"+feat+".png")
            can3.Close()
            can4 = TCanvas(feat+"c4_"+str(particle), feat+"_"+str(particle))
            can4.SetLogy()
            mainPad4 =  TPad("mainPad4", "top4",    0.0, 0.37, 1.0, 1.00)
            ratioPad4 = TPad("ratioPad4","bottom4", 0.0, 0.02, 1.0, 0.37)
            mainPad4.SetLogy()
            mainPad4.SetBottomMargin(0.01)
            ratioPad4.SetTopMargin(0.05)
            ratioPad4.SetBottomMargin(0.35)
            mainPad4.Draw()
            mainPad4.cd()
            histPa.GetXaxis().SetTitle("Probability")
            histPa.SetLineColor(ROOT.kOrange-3)
            histPa.GetYaxis().SetTitleSize(0.04)
            histPa.GetYaxis().SetLabelSize(0.02)
            histPa.GetXaxis().SetLabelSize(0.02)
            histPa.DrawCopy("hist")
            histPb.GetXaxis().SetTitle("Probability")
            histPb.SetLineColor(ROOT.kAzure+7)
            histPb.GetYaxis().SetTitleSize(0.04)
            histPb.GetYaxis().SetLabelSize(0.02)
            histPb.GetXaxis().SetLabelSize(0.02)
            histPb.DrawCopy("hist SAME")
            legend4 = TLegend(0.68,0.65,0.95,0.85)
            legend4.AddEntry(histPa, "MC_{a}", "l")
            legend4.AddEntry(histPb, "MC_{b}", "l")
            legend4.SetTextSize(0.045)
            legend4.Draw()
            can4.cd()
            ratioPad4.Draw()
            ratioPad4.cd()
            ratio_baseline_4 = histPa.Clone("ratio_histPa_"+feat+str(particle))
            ratio_baseline_4.Divide(histPb)
            ratio_baseline_4.GetYaxis().SetRangeUser(0.,6)
            ratio_baseline_4.SetFillColorAlpha(0, 0.25)
            ratio_baseline_4.SetMarkerStyle(86)
            ratio_baseline_4.SetMarkerColor(ROOT.kOrange-3)
            ratio_baseline_4.SetMarkerSize(0.7)
            ratio_baseline_4.Draw("hist p SAME")
            ratio_baseline_4.GetXaxis().SetTitleOffset(0.65)
            ratio_baseline_4.GetXaxis().SetTitleSize(0.15)
            ratio_baseline_4.GetXaxis().SetLabelSize(0.04)
            ratio_baseline_4.GetYaxis().SetLabelSize(0.04)
            line4 = TLine(ratio_baseline_4.GetXaxis().GetXmin(),1,ratio_baseline_4.GetXaxis().GetXmax(),1)
            line4.SetLineWidth(2)
            line4.SetLineStyle(2)
            line4.Draw("SAME")
            can4.cd()
            can4.Draw()
            can4.Update()
            can4.SaveAs(cwd+"Probability"+str(particle)+"_"+feat+".png")
            can4.Close()
            can5 = TCanvas(feat+"c5_"+str(particle), feat+"_"+str(particle))
            can5.SetLogy()
            histPa_train.GetXaxis().SetTitle("Probability")
            histPa_train.SetLineColor(ROOT.kAzure+7)
            histPa_train.GetYaxis().SetTitleSize(0.04)
            histPa_train.GetYaxis().SetLabelSize(0.02)
            histPa_train.GetXaxis().SetLabelSize(0.02)
            histPa_train.DrawCopy("hist")
            histPa_test.GetXaxis().SetTitle("Probability")
            histPa_test.SetLineColor(ROOT.kOrange-3)
            histPa_test.GetYaxis().SetTitleSize(0.04)
            histPa_test.GetYaxis().SetLabelSize(0.02)
            histPa_test.GetXaxis().SetLabelSize(0.02)
            histPa_test.DrawCopy("hist same")
            legend5 = TLegend(0.68,0.65,0.95,0.85)
            legend5.AddEntry(histPa_train, "MC_{a} Train", "l")
            legend5.AddEntry(histPa_test, "MC_{a} Test", "l")
            legend5.Draw()
            can5.SaveAs(cwd+"Probability_trainNtest"+str(particle)+"_"+feat+".png")
            can5.Close()
if __name__ == "__main__":
    args = DSNNr.handle_args()
    makingClosurePlots(args)

