# Lightweight Depression Detection Using 3D Facial Landmark Pseudo-Images and CNN-LSTM

## Overview
This repository contains the source code and preprocessing scripts associated with the following article:

Lightweight Depression Detection Using 3D Facial Landmark Pseudo-Images and CNN-LSTM on DAIC-WOZ and E-DAIC  
Authors: [Jallaglag achraf and al. ]  
Journal: Biomedinformatics (MDPI)  
DOI: to be added upon online publication  

The proposed framework investigates video-based depression screening using 3D facial landmarks, transformed into pseudo-image representations and processed via a CNN–LSTM architecture.

## Repository Structure
├── preprocessing/                                                                                                                                                                        
│ ├── extract_landmarks.py                                                                                                                                                                
│ ├── build_pseudo_images.py                                                                                                                                                     
│ └── normalization.py                                                                                                                                                                    
│                                                                                                                                                                                         
├── models/                                                                                                                                                                               
│ ├── cnn_lstm_model.py                                                                                                                                                                   
│ ├── train_model.py  

│ └── losses.py.py

│                                                                                                                                                                                         
├── evaluation/                                                                                                                                                                           
│ ├── metrics.py                                                                                                                                                                          
│ └── evaluate_model.py                                                                                                                                                                   
│                                                                                                                                                                                         
├── configs/                                                                                                                                                                              
│ └── training_config.yaml                                                                                                                                                                
│                                                                                                                                                                                         
├── requirements.txt                                                                                                                                                                      
└── README.md                                                                                                                                                                             


## Method Summary
- Input: 3D facial landmarks extracted from video sequences  
- Representation: Pseudo-image encoding of temporal facial dynamics  
- Model: CNN for spatial feature extraction + LSTM for temporal modeling  
- Task: Binary depression screening (PHQ-8 based)  
- Datasets: DAIC-WOZ and E-DAIC  

This implementation is intentionally lightweight and privacy-preserving, avoiding raw facial images and multimodal inputs.

## Experimental Setup
- Severe class imbalance handled using macro-average F1-score  
- Results reported as mean ± standard deviation across folds  
- No statistical hypothesis testing performed due to dataset size

## Requirements
Python ≥ 3.8  
Install dependencies using:


pip install -r requirements.txt


## How to Run
### 1. Preprocessing


python preprocessing/extract_landmarks.py
python preprocessing/build_pseudo_images.py


### 2. Training


python models/train_model.py


### 3. Evaluation


python evaluation/evaluate_model.py


## Data Availability
Due to privacy and ethical constraints, the DAIC-WOZ and E-DAIC datasets are not redistributed in this repository. Researchers must request access to the datasets from the original providers.

## Reproducibility
All scripts required to reproduce the preprocessing, training, and evaluation pipelines described in the paper are provided. Hyperparameters and configurations are defined in the configs/ directory.

## Citation
If you use this code, please cite:


@article{sabri2026depression,
title = {Lightweight Depression Detection Using 3D Facial Landmark Pseudo-Images and CNN-LSTM},
author = {Jallaglag achraf and al.},
journal = {Biomedinformatics},
year = {2026},
doi = {TO_BE_ADDED}
}


## Disclaimer
This code is provided for research and educational purposes only.  
It is not intended for clinical diagnosis.

## Contact
For questions or issues, please contact:  
achraf.jallaglag@usmbac.ac.ma
