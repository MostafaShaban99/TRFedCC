Hi,

This repository contains the files associated with our paper titled:

"TRFedCC: A Transformer-Based Resilient Federated ‎Learning with Committee Consensus for FDIA ‎Detection in Energy Community-Driven Smart Grids."

The contents are as follows:

myOPF.m
This MATLAB script generates the raw_data.mat file, which contains state measurements collected over a 2-year period. You can easily modify the code to adjust the dimensionality based on your specific needs.

raw_data.mat
The uncompromised dataset generated from the myOPF.m script. It serves as the clean reference data for the study.

single_trans.py
This Python script defines the Transformer model used in our paper. It is trained on a subset of the dataset for each energy community.

x_data.mat and y_data.mat
These files contain the compromised datasets used in our experiments to simulate false data injection attacks (FDIAs).

We hope you find this repository helpful 😊
If you have any questions, feel free to reach out!
