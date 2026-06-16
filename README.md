# Lightweight Depression Detection Using 3D Facial Landmark Pseudo-Images and CNN-LSTM

## Overview

This repository contains the source code and preprocessing scripts associated with the following published article:

**Lightweight Depression Detection Using 3D Facial Landmark Pseudo-Images and CNN-LSTM on DAIC-WOZ and E-DAIC**

**Authors:** Achraf Jallaglag, My Abdelouahed Sabri, Ali Yahyaouy, and Abdellah Aarab  
**Journal:** *BioMedInformatics* (MDPI)  
**Volume:** 6(1), 2026  
**DOI:** https://doi.org/10.3390/biomedinformatics6010008

The proposed framework investigates video-based depression screening using 3D facial landmarks transformed into pseudo-image representations and processed using a CNN–LSTM architecture.

---

## Repository Structure

```text
├── preprocessing/
│   ├── extract_landmarks.py
│   ├── build_pseudo_images.py
│   └── normalization.py
│
├── models/
│   ├── cnn_lstm_model.py
│   ├── train_model.py
│   └── losses.py
│
├── evaluation/
│   ├── metrics.py
│   └── evaluate_model.py
│
├── configs/
│   └── training_config.yaml
│
├── requirements.txt
└── README.md
```

---

## Method Summary

- **Input:** 3D facial landmarks extracted from video sequences.
- **Representation:** Pseudo-image encoding of temporal facial dynamics.
- **Model:** CNN for spatial feature extraction followed by LSTM for temporal modeling.
- **Task:** Binary depression screening based on PHQ-8 labels.
- **Datasets:** DAIC-WOZ and E-DAIC.

This implementation is intentionally lightweight and privacy-preserving, avoiding the use of raw facial images and multimodal inputs.

---

## Experimental Setup

- Severe class imbalance is handled using the macro-average F1-score.
- Results are reported as mean ± standard deviation across folds.
- No statistical hypothesis testing was performed due to the limited dataset size.

---

## Requirements

Python ≥ 3.8

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Preprocessing

```bash
python preprocessing/extract_landmarks.py
python preprocessing/build_pseudo_images.py
```

### 2. Training

```bash
python models/train_model.py
```

### 3. Evaluation

```bash
python evaluation/evaluate_model.py
```

---

## Data Availability

Due to privacy and ethical constraints, the DAIC-WOZ and E-DAIC datasets are not redistributed in this repository. Researchers should obtain access to the datasets from their original providers.

---

## Reproducibility

All scripts required to reproduce the preprocessing, training, and evaluation pipelines described in the paper are provided. Hyperparameters and configurations are defined in the `configs/` directory.

---

## Citation

If you use this code, please cite:

```bibtex
@article{jallaglag2026depression,
  title={Lightweight Depression Detection Using 3D Facial Landmark Pseudo-Images and CNN-LSTM on DAIC-WOZ and E-DAIC},
  author={Jallaglag, Achraf and Sabri, My Abdelouahed and Yahyaouy, Ali and Aarab, Abdellah},
  journal={BioMedInformatics},
  volume={6},
  number={1},
  year={2026},
  doi={10.3390/biomedinformatics6010008}
}
```

---

## Disclaimer

This code is provided for research and educational purposes only.

It is **not intended for clinical diagnosis**.

---

## Contact

**Achraf Jallaglag**  
Faculty of Sciences Dhar El Mahraz, Sidi Mohamed Ben Abdellah University, Fez, Morocco

📧 **achraf.jallaglag@usmba.ac.ma**
