import numpy as np

def  PreScaleInputRange(MCa, MCb, features, type="linear"):
    # Extract maximum and minimum value of each 
    max = []
    min = []
    for index,feat in enumerate(features):
        # # skip pdgid
        if feat == "pdgid" or feat == "charge":
            continue
        MCa[MCa==-99.]=np.nan
        MCb[MCb==-99.]=np.nan
        feature_array_a = MCa[:,:,index].flatten()
        feature_array_b = MCb[:,:,index].flatten()
        print("feature_array_a:{}, feature_array_b:{}".format(feature_array_a, feature_array_b))
        if type == "linear":
            
            max_a = np.nanmax(feature_array_a)
            max_b = np.nanmax(feature_array_b)
            min_a = np.nanmin(feature_array_a)
            min_b = np.nanmin(feature_array_b)
            #print("max_a:{}, min_a:{}".format(str(max_a),str(min_a)))
            #print("max_b:{}, min_b:{}".format(str(max_b),str(min_b)))
           
            max.append(max_a if max_a > max_b else max_b)
            min.append(min_a if min_a < min_b else min_b)
            
            print("PreScaleInputs::  feature = {}",format(feat))
            print("PreScaleInputs::      max = {}",format(max[-1]))
            print("PreScaleInputs::      min = {}",format(min[-1]))
            '''
            MCaSorted = np.sort(feature_array_a)
            MCbSorted = np.sort(feature_array_b)
            print("MCa sorted:", format(MCaSorted.tolist()))
            print("MCb sorted:", format(MCbSorted.tolist()))
            #print("PreScaleInputs::      max Sherpa:{}, max MGPy8:{}",format(str(max_a), str(max_b)))
            #print("PreScaleInputs::      min Sherpa:{}, min MGPy8:{}",format(str(min_a), str(min_b)))
            MCa[:,:,index] = MCa[:,:,index]-min_a
            MCa[:,:,index] = MCa[:,:,index]/(max_a-min_a)
            MCb[:,:,index] = MCb[:,:,index]-min_b
            MCb[:,:,index] = MCb[:,:,index]/(max_b-min_b)
            '''
            
            MCa[:,:,index] =MCa[:,:,index]-min[-1]
            MCa[:,:,index] =MCa[:,:,index]/(max[-1]-min[-1])

            MCb[:,:,index] =MCb[:,:,index]-min[-1]
            MCb[:,:,index] =MCb[:,:,index]/(max[-1]-min[-1])

            print("PreScaleInputs: MCa   {}, size:{}".format(MCa[:,:,index], len(MCa)))
            print("PreScaleInputs: MCb   {}, size:{}".format(MCb[:,:,index], len(MCb)))
    
    MCa = np.nan_to_num(MCa, nan=-99.)
    MCb = np.nan_to_num(MCb, nan=-99.)
    return MCa,MCb

if __name__ == "__main__":
    path = '/hpcgpfs01/scratch/ftsai/DSNNrBranch/'
    
    Sh1_MCa = np.load(path+'Sherpa_MCa_ptCut.hadd.npz')["MCa"]
    MG1_MCb = np.load(path+'MGPy8_MCa_ptCut.hadd.npz')["MCb"]
    Sh2_MCa = np.load(path+'Sherpa_MCd_ptCut.hadd.npz')["MCa"] 
    MG2_MCb = np.load(path+'MGPy8_MCd_ptCut.hadd.npz')["MCb"]
    Sh3_MCa = np.load(path+'Sherpa_MCe_ptCut.hadd.npz')["MCa"]
    MG3_MCb = np.load(path+'MGPy8_MCe_ptCut.hadd.npz')["MCb"]
    MCa = np.array([*Sh1_MCa, *Sh2_MCa, *Sh3_MCa])
    MCb = np.array([*MG1_MCb, *MG2_MCb, *MG3_MCb])
    Sh1_Wa = np.load(path+'Sherpa_MCa_ptCut.hadd.npz')["MCa_weights"]
    MG1_Wb = np.load(path+'MGPy8_MCa_ptCut.hadd.npz')["MCb_weights"]
    Sh2_Wa = np.load(path+'Sherpa_MCd_ptCut.hadd.npz')["MCa_weights"]
    MG2_Wb = np.load(path+'MGPy8_MCd_ptCut.hadd.npz')["MCb_weights"]
    Sh3_Wa = np.load(path+'Sherpa_MCe_ptCut.hadd.npz')["MCa_weights"]
    MG3_Wb = np.load(path+'MGPy8_MCe_ptCut.hadd.npz')["MCb_weights"]
    MCa_weights = np.array([*Sh1_Wa, *Sh2_Wa, *Sh3_Wa])
    MCb_weights = np.array([*MG1_Wb, *MG2_Wb, *MG3_Wb])
    Sh1_Sa = np.load(path+'Sherpa_MCa_ptCut.hadd.npz')["MCa_spec"]
    MG1_Sb = np.load(path+'MGPy8_MCa_ptCut.hadd.npz')["MCb_spec"]
    Sh2_Sa = np.load(path+'Sherpa_MCd_ptCut.hadd.npz')["MCa_spec"]
    MG2_Sb = np.load(path+'MGPy8_MCd_ptCut.hadd.npz')["MCb_spec"]
    Sh3_Sa = np.load(path+'Sherpa_MCe_ptCut.hadd.npz')["MCa_spec"]
    MG3_Sb = np.load(path+'MGPy8_MCe_ptCut.hadd.npz')["MCb_spec"]
    MCa_spec = np.array([*Sh1_Sa, *Sh2_Sa, *Sh3_Sa])
    MCb_spec = np.array([*MG1_Sb, *MG2_Sb, *MG3_Sb])
    
    features = [
            "pt",
            "eta",
            "phi",
            "m",
            "pdgid",
        ]
    MCa_total, MCb_total = PreScaleInputRange(MCa, MCb, features, "linear")
    arrays = [MCa_total, MCb_total]
    indices = [2, 3, 4]  # Indices of the arrays to transform: leading, sub-leading, and the third jets.
    # PDGIDs for b is 4, for c is 5 and for light quark is 0.
    column = 4  # ID column to transform within the selected indices

    for array in arrays:
        for index in indices:
            array[:, index, column] = [np.nan if value == -99 else value * 0.1 for value in array[:, index, column]]

    np.savez("SherpaOutputs", MCa=MCa_total, MCa_weights=MCa_weights, MCa_spec=MCa_spe)
    np.savez("MGPy8Outputs", MCb=MCb_total, MCb_weights=MCb_weights, MCb_spec=MCb_spec)




