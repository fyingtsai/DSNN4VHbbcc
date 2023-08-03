from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import argparse
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import DSNNr_5050 as DSNNr
from collections import defaultdict
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
# Global plot settings
from matplotlib import rc
import matplotlib.font_manager
from calibration import HistogramCalibrator

from ROOT import TH2F, TCanvas, TH1F, gROOT, gStyle, TLegend, TLorentzVector, TPad, TLine, gPad, TGraphErrors, TLatex, TFile
import ROOT as ROOT
import root_numpy as rn
import math
from numpy import argmax
from array import array
from numpy import load
rc("font", family="serif")
rc("text", usetex=False)
rc("font", size=22)
rc("xtick", labelsize=15)
rc("ytick", labelsize=15)
rc("legend", fontsize=7.5)

ROOT.PyConfig.IgnoreCommandLineOptions = True  # stop PyRoot hijacking -h WHY DOESNT THIS ALWAYS WORK
ROOT.gROOT.SetBatch(True)  # Don't want to plot these to screen as we generate them
ROOT.gROOT.SetStyle('ATLAS')
# Remove stat box
gStyle.SetOptStat(0)
gStyle.SetTitleFontSize(0.08)
gStyle.SetLabelSize(0.1)

model = keras.models.load_model("/hpcgpfs01/scratch/ftsai/DSNNrBranch/ImprovingTraining/saved_models_fix3rdJet_P110105100F95_7layers/model-055.ckpt")
#MCa
#mBB,pTV,dRBB,nJ,dEtaBB,dPhiVBB,dPhiLBmin,dYWH,mTW,pTB1,pTB2,Mtop,met_met,nTags,sumPtJets,HTBoosted,HT_bdtr
obs = ["mBB","pTV","dRBB","nJ","dEtaBB"]
obs_axis = ["m(jet,jet) [GeV]","pTV [GeV]","dR(jet,jet)","nJ","dEta(jet,jet)"]

# obs_idx = 0
inputMCaFile = '/hpcgpfs01/scratch/ftsai/DSNNrBranch/ImprovingTraining/SherpaOutputs.npz'
MCa = np.load(inputMCaFile)["MCa"]
eventRunA=len(MCa)
MCa_spec = np.load(inputMCaFile)["MCa_spec"]
MCa_weights = np.load(inputMCaFile)["MCa_weights"]
#MCb
inputMCbFile = '/hpcgpfs01/scratch/ftsai/DSNNrBranch/ImprovingTraining/MGPy8Outputs.npz'
MCb = np.load(inputMCbFile)["MCb"]
eventRun = len(MCb)
MCb_spec = np.load(inputMCbFile)["MCb_spec"]
MCb_weights = np.load(inputMCbFile)["MCb_weights"]
features = ['pt','eta','phi','m']

PATHtoSplitFile = '/hpcgpfs01/scratch/ftsai/DSNNr4gpu/DSNNDemo/'
with np.load(PATHtoSplitFile+'DataMCadeSplit5050jetFlav_ID.npz', allow_pickle=True, mmap_mode='r') as data3:
   spec_test = data3["S_test"]
   Y_test = data3["Y_test"]
   Y_train = data3["Y_train"]
   W_test = data3["W_test"]
   W_train = data3["W_train"]
   X_test = data3["X_test"]
   X_train = data3["X_train"]
   Y_train = argmax(Y_train, axis=1)
   Y_test = argmax(Y_test,axis=1)

Y_test = argmax(Y_test,axis=1)
Y_train = argmax(Y_train, axis=1)

for obs_idx in range(0, 2):
    MCa_spec_mBB = MCa_spec[:,obs_idx][:eventRunA] #mBB,pTV,dRBB,nJ,dEtaBB
    MCa_weights = MCa_weights[:eventRunA]
    MCa_spec_nJ = MCa_spec[:,3][:eventRunA]
    
    MCa_spec = MCa_spec[:eventRunA]
    MCa_ntags = MCa_spec[:,13][:eventRunA]
    filter_2J_MCa = np.logical_and(MCa_spec_nJ==2, MCa_ntags==2)
    filter_3J_MCa = np.logical_and(MCa_spec_nJ==3, MCa_ntags==2)

    MCb_spec_mBB = MCb_spec[:,obs_idx][:eventRun]
    MCb_weights = MCb_weights[:eventRun]
    MCb_spec_nJ = MCb_spec[:,3][:eventRun]

    MCb_spec = MCb_spec[:eventRun]
    MCb_ntags = MCb_spec[:,13][:eventRun]
    filter_MCb = MCb_spec_mBB[MCb_ntags==2]
    filter_2J_MCb = np.logical_and(MCb_spec_nJ==2, MCb_ntags==2)
    filter_3J_MCb = np.logical_and(MCb_spec_nJ==3, MCb_ntags==2)

    #spec_train_MCa_mBB = spec_train[:,0][Y_train == 0]
    #print("spec_train_MCa_mBB:{}".format(spec_train_MCa_mBB))
    spec_test_MCa_mBB_all = spec_test[:,obs_idx][Y_test==0]
    filter_test_mBB = spec_test[:,obs_idx][Y_test==0]
    #filter_test_2jets_mBB = np.logical_and(np.logical_and(spec_test[:,obs_idx][Y_test==0], spec_test[:,3][Y_test==0]==2), spec_test[:,13][Y_test==0]==2)
    #filter_test_3jets_mBB = np.logical_and(np.logical_and(spec_test[:,obs_idx][Y_test==0], spec_test[:,3][Y_test==0]==3), spec_test[:,13][Y_test==0]==2)
    filter_test_2jets_mBB = np.logical_and(spec_test[:,obs_idx][Y_test==0], spec_test[:,3][Y_test==0]==2)
    filter_test_3jets_mBB = np.logical_and(spec_test[:,obs_idx][Y_test==0], spec_test[:,3][Y_test==0]==3)
    spec_test_MCa_mBB = spec_test[:,obs_idx][Y_test==0]
    spec_test_2jets_mBB = spec_test_MCa_mBB_all[filter_test_2jets_mBB]
    spec_test_3jets_mBB = spec_test_MCa_mBB_all[filter_test_3jets_mBB]
    
    X_train_MCa = X_train[Y_train == 0]
    X_train_MCb = X_train[Y_train == 1]
    X_test_MCa = X_test[Y_test == 0]
    X_test_MCb = X_test[Y_test == 1]
    train_weights_MCa = W_train[Y_train == 0]
    test_weights_MCa = W_test[Y_test == 0]
    preds_train = model.predict(X_train_MCa)
    preds_train_b = model.predict(X_train_MCb)
    preds_test_b = model.predict(X_test_MCb)
    NNweights_train = preds_train[:, 1] / preds_train[:, 0]
    preds_test = model.predict(X_test_MCa)
    #preds_test_b = model.predict(X_test_MCb)
    calibrator = HistogramCalibrator(preds_train[:, 0], preds_train_b[:, 0])
    #calibrator = HistogramCalibrator(preds_train[:, 0], preds_train_b[:, 0], histrange=[0,1])
    Prob_MCa_test = preds_test[:, 0].flatten()
    #Prob_MCa_test = Prob_MCa_test.tolist()
    Cali_PA, Cali_a = calibrator.cali_pred(preds_test[:, 0])
    Cali_PA = np.ma.masked_invalid(Cali_PA)
    Cali_a = np.ma.masked_invalid(Cali_a)
    #NNweights_test = preds_test[:, 1] / preds_test[:, 0]
    NNweights_test = Cali_a/Cali_PA
    #weightTrainR_expand = NNweights_train * train_weights_MCa
    weightTestR_expand = NNweights_test * test_weights_MCa
    weightTestR_expand_2jets = NNweights_test[filter_test_2jets_mBB] * test_weights_MCa[filter_test_2jets_mBB]
    weightTestR_expand_3jets = NNweights_test[filter_test_3jets_mBB] * test_weights_MCa[filter_test_3jets_mBB]
    
    nBins = 20
    if obs_idx == 0:
        minElem = 25
        maxElem = 500
    if obs_idx == 1:
        minElem = 0
        maxElem = 2000 
    if obs_idx == 2: 
        minElem = 0.5
        maxElem = 6 
    #maxElem = np.amax(spec_train_MCa_mBB)
    histMbb_MCa = TH1F("mBB_MCa", "mBB_MCa", nBins, minElem, maxElem)
    histMbb_MCb = TH1F("mBB_MCb", "mBB_MCb", nBins, minElem, maxElem)
    trainMbb_rMCa = TH1F("mBB_train", "mBB_train", nBins, minElem, maxElem)
    testMbb_rMCa = TH1F("mBB_test", "mBB_test", nBins, minElem, maxElem)
    allJet2D = TH2F("allJet2D", "observable", 100, 0.32, 0.85, 100, 0.32, 0.85)
    #maxElem_2 = np.amax(MCb_spec_mBB[MCb_spec_nJ==2])
    histMbb_MCa_2jets = TH1F("mBB_MCa_2jets", "mBB_MCa_2jets", nBins, minElem, maxElem)
    histMbb_MCb_2jets = TH1F("mBB_MCb_2jets", "mBB_MCb_2jets", nBins, minElem, maxElem)
    testMbb_rMCa_2jets = TH1F("mBB_test_2jets", "mBB_test_2jets", nBins, minElem, maxElem)
    
    #maxElem_3 = np.amax(MCb_spec_mBB[MCb_spec_nJ==3])
    histMbb_MCa_3jets = TH1F("mBB_MCa_3jets", "mBB_MCa_3jets", nBins, minElem, maxElem)
    histMbb_MCb_3jets = TH1F("mBB_MCb_3jets", "mBB_MCb_3jets", nBins, minElem, maxElem)
    testMbb_rMCa_3jets = TH1F("mBB_test_3jets", "mBB_test_3jets", nBins, minElem, maxElem)
    if obs_idx == 0:
        binNum = 17
        #newbins_1 = [25,35,45,55,75,125,200,300,400,500]
        newbins_1 = [25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,450,500]
    if obs_idx == 1:
        binNum = 7 
        newbins_1 = [0,100,125,150,200,300,400,2000]
    if obs_idx == 2:
        binNum = 16
        newbins_1 = [0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,5.0,6.0]
    
    newbins_array_1 = array('d', newbins_1)
    histMbb_MCa = histMbb_MCa.Rebin(binNum, "rebinned_mBB_MCa", newbins_array_1)
    histMbb_MCb = histMbb_MCb.Rebin(binNum, "rebinned_mBB_MCb", newbins_array_1)
    testMbb_rMCa = testMbb_rMCa.Rebin(binNum, "rebinned_mBB_test", newbins_array_1)
    histMbb_MCa_2jets = histMbb_MCa_2jets.Rebin(binNum, "rebinned_MCa_2jets", newbins_array_1)
    histMbb_MCb_2jets = histMbb_MCb_2jets.Rebin(binNum, "rebinned_MCb_2jets", newbins_array_1)
    testMbb_rMCa_2jets = testMbb_rMCa_2jets.Rebin(binNum, "rebinned_test_2jets", newbins_array_1)
    histMbb_MCa_3jets = histMbb_MCa_3jets.Rebin(binNum, "rebinned_MCa_3jets", newbins_array_1)
    histMbb_MCb_3jets = histMbb_MCb_3jets.Rebin(binNum, "rebinned_MCb_3jets", newbins_array_1)
    testMbb_rMCa_3jets = testMbb_rMCa_3jets.Rebin(binNum, "rebinned_test_3jets", newbins_array_1)
    
    ##all jets
    rn.fill_hist(histMbb_MCa, MCa_spec_mBB, MCa_weights)
    rn.fill_hist(histMbb_MCb, MCb_spec_mBB, MCb_weights)
    #rn.fill_hist(trainMbb_rMCa, spec_train_MCa_mBB, weightTrainR_expand)
    rn.fill_hist(testMbb_rMCa, spec_test_MCa_mBB, weightTestR_expand)
    ### Fill a 2D plot
    #data = np.column_stack((Cali_PA, Prob_MCa_test))
    #data = np.column_stack((Cali_PA, spec_test_MCa_mBB * weightTestR_expand))
    #rn.fill_hist(allJet2D, data)
    #canvas = TCanvas("canvas", "2D Histogram", 800, 600)
    #canvas.SetLogx()
    #allJet2D.Draw("colz")
    #canvas.SaveAs("2Dplot.pdf")
    ##2 jets
    rn.fill_hist(histMbb_MCa_2jets, MCa_spec_mBB[MCa_spec_nJ==2], MCa_weights[MCa_spec_nJ==2])
    rn.fill_hist(histMbb_MCb_2jets, MCb_spec_mBB[MCb_spec_nJ==2], MCb_weights[MCb_spec_nJ==2])
    rn.fill_hist(testMbb_rMCa_2jets, spec_test_2jets_mBB, weightTestR_expand_2jets)
    ## 3 jets
    rn.fill_hist(histMbb_MCa_3jets, MCa_spec_mBB[MCa_spec_nJ==3], MCa_weights[MCa_spec_nJ==3])
    rn.fill_hist(histMbb_MCb_3jets, MCb_spec_mBB[MCb_spec_nJ==3], MCb_weights[MCb_spec_nJ==3])
    rn.fill_hist(testMbb_rMCa_3jets, spec_test_3jets_mBB, weightTestR_expand_3jets)
    
    histMbb_MCa.Scale(1/histMbb_MCa.Integral())
    histMbb_MCb.Scale(1/histMbb_MCb.Integral())
    #trainMbb_rMCa.Scale(1/trainMbb_rMCa.Integral())
    testMbb_rMCa.Scale(1/testMbb_rMCa.Integral())
    
    histMbb_MCa_2jets.Scale(1/histMbb_MCa_2jets.Integral())
    histMbb_MCb_2jets.Scale(1/histMbb_MCb_2jets.Integral())
    testMbb_rMCa_2jets.Scale(1/testMbb_rMCa_2jets.Integral())
    histMbb_MCa_3jets.Scale(1/histMbb_MCa_3jets.Integral())
    histMbb_MCb_3jets.Scale(1/histMbb_MCb_3jets.Integral())
    testMbb_rMCa_3jets.Scale(1/testMbb_rMCa_3jets.Integral())
    
    can1 = TCanvas("mBB","mBB")
    mainPad =  TPad("mainPad", "top",    0.02, 0.45, 0.98, 0.98)
    ratioPad = TPad("ratioPad","bottom", 0.02, 0.08, 0.98, 0.45) #(,0.02,,)
    mainPad.SetBottomMargin(0.01)
    ratioPad.SetTopMargin(0.04)
    ratioPad.SetBottomMargin(0.15)
    #mainPad.SetLogy()
    mainPad.Draw()
    mainPad.cd()

    histMbb_MCa.GetYaxis().SetTitle("#frac{d#sigma}{d"+obs[obs_idx]+"}")
    histMbb_MCa.SetFillColorAlpha(ROOT.kAzure+7, 0.25)
    histMbb_MCa.SetLineColor(ROOT.kAzure+7)
    histMbb_MCa.GetYaxis().SetTitleSize(0.04)
    histMbb_MCa.GetYaxis().SetLabelSize(0.02)
    histMbb_MCa.GetXaxis().SetLabelSize(0.02)
    histMbb_MCa.GetYaxis().SetTitleOffset(0.7)
    histMbb_MCa.DrawCopy("hist")
    
    histMbb_MCb.GetYaxis().SetTitle("#frac{d#sigma}{d"+obs[obs_idx]+"}")
    histMbb_MCb.SetFillColorAlpha(ROOT.kOrange-3, 0.25)
    histMbb_MCb.SetLineColor(ROOT.kOrange-3)
    histMbb_MCb.GetYaxis().SetTitleSize(0.04)
    histMbb_MCb.GetYaxis().SetLabelSize(0.02)
    histMbb_MCb.GetXaxis().SetLabelSize(0.02)
    histMbb_MCb.GetYaxis().SetTitleOffset(0.7)
    histMbb_MCb.DrawCopy("hist SAME")
    
    testMbb_rMCa.SetFillColorAlpha(ROOT.kBlue, 0.25)
    testMbb_rMCa.SetLineColor(ROOT.kBlue)
    testMbb_rMCa.GetYaxis().SetTitleSize(0.04)
    testMbb_rMCa.GetYaxis().SetLabelSize(0.02)
    testMbb_rMCa.GetXaxis().SetLabelSize(0.02)
    testMbb_rMCa.DrawCopy("hist SAME")
    
    tt1 = ROOT.TLatex()
    tt1.SetNDC()
    tt1.SetTextSize(0.07)
    tt1.DrawLatex(0.35 ,0.75 ,"p-value(test, MCb):"+str(format(testMbb_rMCa.Chi2Test(histMbb_MCb,"WW P"),'.1E')))
    tt2 = ROOT.TLatex()
    tt2.SetNDC()
    tt2.SetTextSize(0.055)
    #tt2.DrawLatex(0.45 ,0.70 ,"p-value(train, MCb):"+str(format(trainMbb_rMCa.Chi2Test(histMbb_MCb,"WW P"),'.1E')))
    tt3 = ROOT.TLatex()
    tt3.SetNDC()
    tt3.SetTextSize(0.07)
    tt3.DrawLatex(0.35 ,0.60 ,"p-value(MCa, MCb):"+str(format(histMbb_MCa.Chi2Test(histMbb_MCb,"WW P"),'.1E')))
    
    legend = TLegend(0.64,0.65,0.95,0.85)
    legend.SetBorderSize(0)
    legend.AddEntry(histMbb_MCb, "MGPy8", "l")
    legend.AddEntry(histMbb_MCa, "Sherpa", "l")
    legend.AddEntry(testMbb_rMCa, "Sherpa(testing) * NN", "l")
    #legend.AddEntry(trainMbb_rMCa, "MCa(training) * NN", "l")
    n_ex=[]
    n_ey=[]
    n_x=[]
    n_y=[]
    
    for hisBin in range(histMbb_MCa.GetNbinsX()+1):
        if hisBin==0:   continue
        n_x.append(histMbb_MCa.GetBinLowEdge(hisBin)+ 0.5*(histMbb_MCa.GetBinLowEdge(hisBin+1)-(histMbb_MCa.GetBinLowEdge(hisBin))))
        n_y.append(1) #center at 1 in the y-axis
        n_ex.append((histMbb_MCa.GetBinLowEdge(hisBin+1)-histMbb_MCa.GetBinLowEdge(hisBin))*0.5)
        if testMbb_rMCa.GetBinContent(hisBin)+histMbb_MCb.GetBinError(hisBin)!=0:
            n_ey.append(math.sqrt(pow(histMbb_MCb.GetBinError(hisBin),2)+pow(testMbb_rMCa.GetBinError(hisBin),2))/(histMbb_MCb.GetBinContent(hisBin)+testMbb_rMCa.GetBinContent(hisBin)))
        else:
            n_ey.append(2)
    
    gr = TGraphErrors(binNum,np.array(n_x, dtype=float),np.array(n_y, dtype=float),np.array(n_ex, dtype=float),np.array(n_ey, dtype=float))
    gr.GetXaxis().SetLimits(minElem,maxElem)
    gr.GetYaxis().CenterTitle(1)
    gr.GetYaxis().SetTitleOffset(0.25)
    gr.GetYaxis().SetTitleSize(0.1)
    gr.GetXaxis().SetTitle(obs_axis[obs_idx])
    gr.GetXaxis().SetTitleOffset(0.55)#0.25
    gr.GetXaxis().SetTitleSize(0.10)
    gr.GetYaxis().SetLabelSize(0.02)
    gr.GetXaxis().SetLabelSize(0.04)
    gr.SetMarkerStyle(1)
    gr.SetFillColorAlpha(ROOT.kPink+6, 0.25)
    gr.SetLineColor(ROOT.kPink+6)
    
    gr.GetYaxis().SetTitle("w.r.t MGPy8")
    gr.GetYaxis().SetRangeUser(0.5,1.5)
    
    legend.AddEntry(gr, "(Test+MCb)Stat Unc.")
    legend.SetTextSize(0.045)
    legend.Draw()
    can1.cd()
    ratioPad.Draw()
    ratioPad.cd()
    gr.Draw("a2")
    line = TLine(0,1,maxElem,1)
    line.SetLineWidth(2)
    line.SetLineStyle(2)
    line.Draw("SAME")
    
    ratio_baseline = histMbb_MCa.Clone("ratio_hista")
    ratio_baseline.Divide(histMbb_MCb)
    ratio_baseline.SetFillColorAlpha(0, 0.25)
    ratio_baseline.SetMarkerStyle(86)
    ratio_baseline.SetMarkerColor(ROOT.kAzure+7)
    ratio_baseline.SetMarkerSize(0.7)
    ratio_baseline2 = testMbb_rMCa.Clone("ratio_histr_test")
    ratio_baseline2.Divide(histMbb_MCb)
    ratio_baseline2.SetMarkerColor(ROOT.kBlue)
    ratio_baseline2.SetMarkerSize(0.7)
    ratio_baseline2.GetYaxis().SetRangeUser(0.5,1.5)
    ratio_baseline2.GetYaxis().CenterTitle(1)
    ratio_baseline2.GetYaxis().SetTitleOffset(0.5)
    ratio_baseline2.GetYaxis().SetTitle("w.r.t MGPy8")
    ratio_baseline2.GetYaxis().SetTitleSize(0.1)
    ratio_baseline2.GetXaxis().SetTitle(obs_axis[obs_idx])
    ratio_baseline2.GetXaxis().SetTitleOffset(0.55) #0.25
    ratio_baseline2.GetXaxis().SetTitleSize(0.15) #0.1
    ratio_baseline2.GetYaxis().SetLabelSize(0.04)
    ratio_baseline2.Draw("hist p SAME")
    ratio_baseline.Draw("hist p SAME")
    myfile = TFile('h_'+obs[obs_idx]+'_jflav_test7.root', 'RECREATE')
    ratio_baseline.Write()
    ratio_baseline2.Write()
    myfile.Close()
    can1.cd()
    can1.Draw()
    can1.Update()
    can1.SaveAs("Spec/Obs_"+obs[obs_idx]+"_Demo.png")
    can1.Close()
    
    
    can2 = TCanvas("mBB_2jet","mBB")
    mainPad_2 =  TPad("mainPad_2", "top_2",    0.02, 0.45, 0.98, 0.98)
    ratioPad_2 = TPad("ratioPad_2","bottom_2", 0.02, 0.08, 0.98, 0.45)
    mainPad_2.SetBottomMargin(0.01)
    ratioPad_2.SetTopMargin(0.04)
    ratioPad_2.SetBottomMargin(0.15)
    
    
    #mainPad_2.SetLogy()
    mainPad_2.Draw()
    mainPad_2.cd()
    
    histMbb_MCa_2jets.GetYaxis().SetTitle("#frac{d#sigma}{"+obs[obs_idx]+"}")
    histMbb_MCa_2jets.SetFillColorAlpha(ROOT.kAzure+7, 0.25)
    histMbb_MCa_2jets.SetLineColor(ROOT.kAzure+7)
    histMbb_MCa_2jets.GetYaxis().SetTitleSize(0.04)
    histMbb_MCa_2jets.GetYaxis().SetLabelSize(0.02)
    histMbb_MCa_2jets.GetXaxis().SetLabelSize(0.02)
    histMbb_MCa_2jets.GetYaxis().SetTitleOffset(0.7)
    histMbb_MCa_2jets.DrawCopy("hist")
    
    histMbb_MCb_2jets.GetYaxis().SetTitle("#frac{d#sigma}{d"+obs[obs_idx]+"}")
    histMbb_MCb_2jets.SetFillColorAlpha(ROOT.kOrange-3, 0.25)
    histMbb_MCb_2jets.SetLineColor(ROOT.kOrange-3)
    histMbb_MCb_2jets.GetYaxis().SetTitleSize(0.04)
    histMbb_MCb_2jets.GetYaxis().SetLabelSize(0.02)
    histMbb_MCb_2jets.GetXaxis().SetLabelSize(0.02)
    histMbb_MCb_2jets.GetYaxis().SetTitleOffset(0.7)
    histMbb_MCb_2jets.DrawCopy("hist SAME")
    
    testMbb_rMCa_2jets.SetFillColorAlpha(ROOT.kBlue, 0.25)
    testMbb_rMCa_2jets.SetLineColor(ROOT.kBlue)
    testMbb_rMCa_2jets.GetYaxis().SetTitleSize(0.04)
    testMbb_rMCa_2jets.GetYaxis().SetLabelSize(0.02)
    testMbb_rMCa_2jets.GetXaxis().SetLabelSize(0.02)
    testMbb_rMCa_2jets.DrawCopy("hist SAME")
    
    tt1 = ROOT.TLatex()
    tt1.SetNDC()
    tt1.SetTextSize(0.055)
    tt1.DrawLatex(0.35 ,0.75 ,"p-value(test, MCb):"+str(format(testMbb_rMCa_2jets.Chi2Test(histMbb_MCb_2jets,"WW P"),'.1E')))
    tt2 = ROOT.TLatex()
    tt2.SetNDC()
    tt2.SetTextSize(0.055)
    #tt2.DrawLatex(0.45 ,0.70 ,"p-value(train, MCb):"+str(format(trainMbb_rMCa.Chi2Test(histMbb_MCb,"WW P"),'.1E')))
    tt3 = ROOT.TLatex()
    tt3.SetNDC()
    tt3.SetTextSize(0.055)
    tt3.DrawLatex(0.35 ,0.65 ,"p-value(MCa, MCb):"+str(format(histMbb_MCa_2jets.Chi2Test(histMbb_MCb_2jets,"WW P"),'.1E')))
    
    legend = TLegend(0.64,0.65,0.95,0.85)
    legend.SetBorderSize(0)
    legend.AddEntry(histMbb_MCb_2jets, "MGPy8 (2jets)", "l")
    legend.AddEntry(histMbb_MCa_2jets, "Sherpa (2jets)", "l")
    legend.AddEntry(testMbb_rMCa_2jets, "Sherpa(testing_2jets) * NN", "l")
    #legend.AddEntry(trainMbb_rMCa, "MCa(training) * NN", "l")
    n_ex=[]
    n_ey=[]
    n_x=[]
    n_y=[]
    
    for hisBin in range(histMbb_MCa_2jets.GetNbinsX()+1):
        if hisBin==0:   continue
        n_x.append(histMbb_MCa_2jets.GetBinLowEdge(hisBin)+ 0.5*(histMbb_MCa_2jets.GetBinLowEdge(hisBin+1)-(histMbb_MCa_2jets.GetBinLowEdge(hisBin))))
        n_y.append(1) #center at 1 in the y-axis
        n_ex.append((histMbb_MCa_2jets.GetBinLowEdge(hisBin+1)-histMbb_MCa_2jets.GetBinLowEdge(hisBin))*0.5)
        if testMbb_rMCa_2jets.GetBinContent(hisBin)+histMbb_MCb_2jets.GetBinError(hisBin)!=0:
            n_ey.append(math.sqrt(pow(histMbb_MCb_2jets.GetBinError(hisBin),2)+pow(testMbb_rMCa_2jets.GetBinError(hisBin),2))/(histMbb_MCb_2jets.GetBinContent(hisBin)+testMbb_rMCa_2jets.GetBinContent(hisBin)))
        else:
            n_ey.append(2)
    
    gr = TGraphErrors(binNum,np.array(n_x, dtype=float),np.array(n_y, dtype=float),np.array(n_ex, dtype=float),np.array(n_ey, dtype=float))
    gr.GetXaxis().SetLimits(minElem,maxElem)
    gr.GetYaxis().CenterTitle(1)
    gr.GetYaxis().SetTitleOffset(0.3)
    gr.GetYaxis().SetTitleSize(0.1)
    gr.GetXaxis().SetTitle(obs_axis[obs_idx])
    gr.GetXaxis().SetTitleOffset(0.55)#0.25
    gr.GetXaxis().SetTitleSize(0.10)
    gr.GetYaxis().SetLabelSize(0.02)
    gr.GetXaxis().SetLabelSize(0.04)
    gr.SetMarkerStyle(1)
    gr.SetFillColorAlpha(ROOT.kPink+6, 0.25)
    gr.SetLineColor(ROOT.kPink+6)
    
    gr.GetYaxis().SetTitle("w.r.t MGPy8")
    gr.GetYaxis().SetRangeUser(0.5,1.5)
    
    legend.AddEntry(gr, "(Test+MCb)Stat Unc.")
    legend.SetTextSize(0.045)
    legend.Draw()
    can2.cd()
    ratioPad_2.Draw()
    ratioPad_2.cd()
    gr.Draw("a2")
    line = TLine(0,1,maxElem,1)
    line.SetLineWidth(2)
    line.SetLineStyle(2)
    line.Draw("SAME")
    ratio_baseline = histMbb_MCa_2jets.Clone("ratio_hista")
    ratio_baseline.Divide(histMbb_MCb_2jets)
    ratio_baseline.SetFillColorAlpha(0, 0.25)
    ratio_baseline.SetMarkerStyle(86)
    ratio_baseline.SetMarkerColor(ROOT.kAzure+7)
    ratio_baseline.SetMarkerSize(0.7)
    ratio_baseline2 = testMbb_rMCa_2jets.Clone("ratio_histr_test")
    ratio_baseline2.Divide(histMbb_MCb_2jets)
    ratio_baseline2.SetMarkerColor(ROOT.kBlue)
    ratio_baseline2.SetMarkerSize(0.7)
    ratio_baseline2.GetYaxis().SetRangeUser(0.5,1.5)
    ratio_baseline2.GetYaxis().CenterTitle(1)
    ratio_baseline2.GetYaxis().SetTitleOffset(0.5)
    ratio_baseline2.GetYaxis().SetTitle("w.r.t MGPy8")
    ratio_baseline2.GetYaxis().SetTitleSize(0.1)
    ratio_baseline2.GetXaxis().SetTitle(obs_axis[obs_idx])
    ratio_baseline2.GetXaxis().SetTitleOffset(0.25)
    ratio_baseline2.GetXaxis().SetTitleSize(0.1)
    ratio_baseline2.GetYaxis().SetLabelSize(0.04)
    ratio_baseline2.Draw("hist p SAME")
    ratio_baseline.Draw("hist p SAME")
    myfile2 = TFile('h_'+obs[obs_idx]+'_2jets_jflav_test7.root', 'RECREATE')
    ratio_baseline.Write()
    ratio_baseline2.Write()
    myfile2.Close()
    can2.cd()
    can2.Draw()
    can2.Update()
    can2.SaveAs("Spec/Obs_"+obs[obs_idx]+"_2jets_Demo.png")
    can2.Close()
    
    can3 = TCanvas("mBB_3jet","mBB")
    mainPad_3 =  TPad("mainPad_3", "top_3",    0.02, 0.45, 0.98,0.98)
    ratioPad_3 = TPad("ratioPad_3","bottom_3", 0.02, 0.08, 0.98, 0.45)
    mainPad_3.SetBottomMargin(0.01)
    ratioPad_3.SetTopMargin(0.08)
    ratioPad_3.SetBottomMargin(0.15)
    #mainPad_3.SetLogy()
    mainPad_3.Draw()
    mainPad_3.cd()
    
    histMbb_MCa_3jets.GetYaxis().SetTitle("#frac{d#sigma}{d"+obs[obs_idx]+"}")
    histMbb_MCa_3jets.SetFillColorAlpha(ROOT.kAzure+7, 0.25)
    histMbb_MCa_3jets.SetLineColor(ROOT.kAzure+7)
    histMbb_MCa_3jets.GetYaxis().SetTitleSize(0.04)
    histMbb_MCa_3jets.GetYaxis().SetLabelSize(0.02)
    histMbb_MCa_3jets.GetXaxis().SetLabelSize(0.02)
    histMbb_MCa_3jets.GetYaxis().SetTitleOffset(0.7)
    histMbb_MCa_3jets.GetXaxis().SetTitleOffset(0.4)
    histMbb_MCa_3jets.DrawCopy("hist")
    
    histMbb_MCb_3jets.GetYaxis().SetTitle("#frac{d#sigma}{d"+obs[obs_idx]+"}")
    histMbb_MCb_3jets.SetFillColorAlpha(ROOT.kOrange-3, 0.25)
    histMbb_MCb_3jets.SetLineColor(ROOT.kOrange-3)
    histMbb_MCb_3jets.GetYaxis().SetTitleSize(0.04)
    histMbb_MCb_3jets.GetYaxis().SetLabelSize(0.02)
    histMbb_MCb_3jets.GetXaxis().SetLabelSize(0.02)
    histMbb_MCb_3jets.GetYaxis().SetTitleOffset(0.7)
    histMbb_MCb_3jets.DrawCopy("hist SAME")
    
    testMbb_rMCa_3jets.SetFillColorAlpha(ROOT.kBlue, 0.25)
    testMbb_rMCa_3jets.SetLineColor(ROOT.kBlue)
    testMbb_rMCa_3jets.GetYaxis().SetTitleSize(0.04)
    testMbb_rMCa_3jets.GetYaxis().SetLabelSize(0.02)
    testMbb_rMCa_3jets.GetXaxis().SetLabelSize(0.02)
    testMbb_rMCa_3jets.DrawCopy("hist SAME")
    
    tt1 = ROOT.TLatex()
    tt1.SetNDC()
    tt1.SetTextSize(0.055)
    tt1.DrawLatex(0.35 ,0.75 ,"p-value(test, MCb):"+str(format(testMbb_rMCa_3jets.Chi2Test(histMbb_MCb_3jets,"WW P"),'.1E')))
    tt3 = ROOT.TLatex()
    tt3.SetNDC()
    tt3.SetTextSize(0.055)
    tt3.DrawLatex(0.35 ,0.65 ,"p-value(MCa, MCb):"+str(format(histMbb_MCa_3jets.Chi2Test(histMbb_MCb_3jets,"WW P"),'.1E')))
    
    legend = TLegend(0.64,0.65,0.95,0.85)
    legend.SetBorderSize(0)
    legend.AddEntry(histMbb_MCb_3jets, "MGPy8 (3jets)", "l")
    legend.AddEntry(histMbb_MCa_3jets, "Sherpa (3jets)", "l")
    legend.AddEntry(testMbb_rMCa_3jets, "Sherpa(testing_3jets) * NN", "l")
    #legend.AddEntry(trainMbb_rMCa, "MCa(training) * NN", "l")
    n_ex=[]
    n_ey=[]
    n_x=[]
    n_y=[]
    
    for hisBin in range(histMbb_MCa_3jets.GetNbinsX()+1):
        if hisBin ==0:  continue
        #print("histMbb_MCa_3jets.GetBinLowEdge(hisBin):{}".format(histMbb_MCa_3jets.GetBinLowEdge(hisBin)))
        n_x.append(histMbb_MCa_3jets.GetBinLowEdge(hisBin)+ 0.5*(histMbb_MCa_3jets.GetBinLowEdge(hisBin+1)-(histMbb_MCa_3jets.GetBinLowEdge(hisBin))))
        n_y.append(1) #center at 1 in the y-axis
        n_ex.append((histMbb_MCa_3jets.GetBinLowEdge(hisBin+1)-histMbb_MCa_3jets.GetBinLowEdge(hisBin))*0.5)
        if testMbb_rMCa_3jets.GetBinContent(hisBin)+histMbb_MCb_3jets.GetBinError(hisBin)!=0:
            n_ey.append(math.sqrt(pow(histMbb_MCb_3jets.GetBinError(hisBin),2)+pow(testMbb_rMCa_3jets.GetBinError(hisBin),2))/(histMbb_MCb_3jets.GetBinContent(hisBin)+testMbb_rMCa_3jets.GetBinContent(hisBin)))
        else:
            n_ey.append(2)
    
    gr = TGraphErrors(binNum,np.array(n_x, dtype=float),np.array(n_y, dtype=float),np.array(n_ex, dtype=float),np.array(n_ey, dtype=float))
    gr.GetXaxis().SetLimits(minElem,maxElem)
    gr.GetYaxis().CenterTitle(1)
    gr.GetYaxis().SetTitleOffset(0.3)
    gr.GetYaxis().SetTitleSize(0.1)
    gr.GetXaxis().SetTitle(obs_axis[obs_idx])
    gr.GetXaxis().SetTitleOffset(0.55)#0.25
    gr.GetXaxis().SetTitleSize(0.10)
    gr.GetYaxis().SetLabelSize(0.02)
    gr.GetXaxis().SetLabelSize(0.04)
    gr.SetMarkerStyle(1)
    gr.SetFillColorAlpha(ROOT.kPink+6, 0.25)
    gr.SetLineColor(ROOT.kPink+6)
    
    gr.GetYaxis().SetTitle("w.r.t MGPy8")
    gr.GetYaxis().SetRangeUser(0.5,1.5)
    
    legend.AddEntry(gr, "(Test+MCb)Stat Unc.")
    legend.SetTextSize(0.045)
    legend.Draw()
    can3.cd()
    ratioPad_3.Draw()
    ratioPad_3.cd()
    gr.Draw("a2")
    line = TLine(0,1,maxElem,1)
    line.SetLineWidth(2)
    line.SetLineStyle(2)
    line.Draw("SAME")
    ratio_baseline = histMbb_MCa_3jets.Clone("ratio_hista")
    ratio_baseline.Divide(histMbb_MCb_3jets)
    ratio_baseline.SetFillColorAlpha(0, 0.25)
    ratio_baseline.SetMarkerStyle(86)
    ratio_baseline.SetMarkerColor(ROOT.kAzure+7)
    ratio_baseline.SetMarkerSize(0.7)
    ratio_baseline2 = testMbb_rMCa_3jets.Clone("ratio_histr_test")
    ratio_baseline2.Divide(histMbb_MCb_3jets)
    ratio_baseline2.SetMarkerColor(ROOT.kBlue)
    ratio_baseline2.SetMarkerSize(0.7)
    ratio_baseline2.GetYaxis().SetRangeUser(0.5,1.5)
    ratio_baseline2.GetYaxis().CenterTitle(1)
    ratio_baseline2.GetYaxis().SetTitleOffset(0.5)
    ratio_baseline2.GetYaxis().SetTitle("w.r.t MGPy8")
    ratio_baseline2.GetYaxis().SetTitleSize(0.1)
    ratio_baseline2.GetXaxis().SetTitle(obs_axis[obs_idx])
    ratio_baseline2.GetXaxis().SetTitleOffset(0.25)
    ratio_baseline2.GetXaxis().SetTitleSize(0.1)
    ratio_baseline2.GetYaxis().SetLabelSize(0.04)
    ratio_baseline2.Draw("hist p SAME")
    ratio_baseline.Draw("hist p SAME")
    myfile3 = TFile('h_'+obs[obs_idx]+'_3jets_jflav_test7.root', 'RECREATE')
    ratio_baseline.Write()
    ratio_baseline2.Write()
    myfile3.Close()
    can3.cd()
    can3.Draw()
    can3.Update()
    can3.SaveAs("Spec/Obs_"+obs[obs_idx]+"_3jets_Demo.png")
    can3.Close()
    
