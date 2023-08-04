# Deep Set Neural Networks 

We learn to implement ["Energy Flow Networks"](https://arxiv.org/abs/1810.05165) for MCs mapping using Keras and TensorFlow (v.2). This repo contains the code samples that are compatible with the CxAOD input formate.

**Project Objectives**:
+ Develop a generalized classification for the VH(bb/cc) analysis through Deep Set neural networks.
+ Test the application running in the various computing environment (HTCondor, SLURM, GCP, HPC...). The project is also built off of an interactive way of running code in Jupyter Notebook.
+ Speed up the data preprocessing and the NN training using DASK, Ray... etc.

**Demo**
We've prepared a demonstration of the DSNN application in a Google Colab notebook. This demo will walk you through the steps of using the DSNN framework for shape systematic uncertainty estimation.
Click [here](https://colab.research.google.com/drive/1YVBuYGpHAc74POLuiqjyiSxwTI6jdZv9#scrollTo=i1vlXcasy1KT) to access the DSNN demo in Google Colab.

In this demo, you'll be able to:

- Explore the data preprocessing steps
- Configure and train the DSNN model
- Visualize the training results
- Apply the trained model to make predictions
- Plot the results using the provided plotting code

**Data Preprocessing**:

For the smaller dataset:
Run the application including preprocessing and training data: `sbatch myscript.sh`<br />

##Note: Data Preprocessing and Campaign Separation: 
The PreScaleInputRange step is performed after data preprocessing, especially to ensure that the input data ranges are appropriately scaled and centered. However running three MC campaings can be CPU-intensive (>250GB) due to the substantial payload involved. Therefore, it is recommended to execute each MC campaign independently. <br />
1. For each campaign, ensure that the necessary adjustments are made by replacing `--MCa` and `--MCb` flags in the `myDSNNr_run.sh` script. Then run the application through: `sbatch myscript.sh` <br />
- Note: The --isTraining flag is a command-line option that can be used when running the train.py script. It is used to indicate whether the script should operate in training mode or not. When the --isTraining flag is provided without a value, it sets the args.isTraining attribute to True, and when it is omitted, the attribute is set to False.<br/>

2. Summation and Data Feature Scaling: Once the three independent MC campaigns are complete, you can proceed to aggregate the results. This involves summing up the data arrays obtained from each campaign. Following this, the step of feature scaling is performed. To do this, using the command `sbatch myscript_feat.sh`. <br />
3. Training inputs preparation: The output of the above step will yield two vital files, namely `SherpaOutputs.npz` and `MGPy8Outputs.npz`. These files serve as essential inputs for training the Deep Set Neural Network (DSNN). <br />
-The split data will be saved as `DataMCaSplit5050_split.npz`. With the split data ready, you can proceed to generate plots later. <br /> 

**Training**

DSNN training: With the properly scaled and centered training inputs, you can now embark on training the DSNN using these dataset. Then execute: `sbatch myscript.sh` with `--isTraining` in the `myDSNNr_run.sh` <br />
##Note: the residual training using EnergyFlow
The EnergyFlow library is downloaded and imported in the DSNNr.py. The file `EnergyFlow/energyflow/archs/efn.py` in the project directory contains the implementation details of the deep-set neural network architecture. <br/>
The `_construct_F` is updated for the 7 layers and connections of the neural networks using the Add() layer from Keras. You will need to experiment the number of residaul connections, layer configurations to find the optimal setup for your specific analysis. <br/>
<br/>
**Generating Plots and Visualizations**
Training using a GPU and the performance plotting are not be able to run together at the moment due to python version conflicts, thus:<br />
Check out the performance: `sbatch plot_Script.sh`<br />

##Note: 
Once the DSNN model is trained and the data is processed, you can proceed to generate plots and visualizations to analyze the model's performance and the impact of shape uncertainties. The plotting code provides flexibility to customize the input data and the observables (spectators) to be plotted. <br/>
1. Load the Trained Model:<br/>
Replace the line model = keras.models.load_model() with the path to your trained DSNN model. This model will be used for making predictions and generating plots.
2. Specify Input Data Paths in the makingSpecPlots.py: <br/>
Replace inputMCaFile and inputMCbFile with the file paths to the processed Sherpa and Madgraph output arrays. <br/>
Replace PATHtoSplitFile with the file path to the split data file generated in the previous step (DataMCaSplit5050_split.npz).<br/>
3. Define Observables (Spectators) in the makingSpecPlots.py: <br/>
Modify the obs list to include the observables (spectators) you would like to plot. For example, obs = ["mBB", "pTV", "dRBB", "nJ", "dEtaBB"] can be adjusted to suit your analysis goals.
4. HistogramCalibrator Implementation:<br/>
Note that a 2-step calibration using the HistogramCalibrator is applied within the makingSpecPlots.py script. This step involves creating a calibration instance with the predictions from both the signal and background components in the makingSpecPlots.py:
calibrator = HistogramCalibrator(preds_train[:, 0], preds_train_b[:, 0])<br/>
You will need to experiment the various probabilities and calibrated regions for your specific analysis.
6. Run the plotting code using `python makingSpecPlots.py`. Submmit the job through: `sbatch plot_Script.sh` <br/>
