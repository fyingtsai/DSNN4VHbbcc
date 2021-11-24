#!/bin/bash

hostname
date
workDir=`pwd`
pwd
conda --version

export fileName='Reco_WJet_MCade_5050'
python3 makingClosurePlots.py --global_name $fileName --features pt,eta,phi,m,pdgid --weightFeature EventWeight --spectators mBB,pTV,dRBB
