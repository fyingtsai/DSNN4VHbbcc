#!/bin/bash

hostname
date
workDir=`pwd`
pwd
conda --version
export fileName='Reco_Wbb_MCa'
python3 train.py --global_name $fileName --MCa $workDir/data/Reco/Test/SherpaMCaWbb.hadd.root --MCb $workDir/data/Reco/Test/MadGraphMCaWbb.hadd.root --features pt,eta,phi,m,pdgid --weightFeature EventWeight --spectators mBB,pTV,dRBB
