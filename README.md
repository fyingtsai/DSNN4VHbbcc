# Deep Set Neural Networks 

We learn to implement ["Energy Flow Networks"](https://arxiv.org/abs/1810.05165) for MCs mapping using Keras and TensorFlow (v.2). This repo contains the code samples that are compatible with the CxAOD input formate.

**Project Objectives**:
+ Develop a generalized classification for the VH(bb/cc) analysis through Deep Set neural networks.
+ Test the application running in the various computing environment (HTCondor, SLURM, GCP, HPC...). The project is also built off of an interactive way of running code in Jupyter Notebook.
+ Speed up the data preprocessing and the NN training using DASK, Ray... etc.

**Submitting a job to SLURM**:

Run the application including preprocessing and training data: `sbatch myscript.sh`<br />
Training using a GPU and the performance plotting are not be able to run together at the moment due to python version conflicts, thus:<br />
Check out the performance: `sbatch plot_Script.sh`<br />
