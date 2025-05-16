# TransGNN-AD: Transductive Graph Neural Networks for Alzheimer's Disease Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of "TransGNN-AD: Transductive Graph Neural Networks with Sparse Labels for Early Alzheimer's Detection from Connectivity Features" (BMVC 2025).

## Overview

TransGNN-AD is a comprehensive framework for Alzheimer's disease (AD) classification using diffusion MRI data through graph neural network architectures in a transductive learning setup. Our approach leverages brain connectivity networks built from tractography-derived fiber counts between regions of interest and engineers histogram-based connectivity features that effectively represent connectivity distribution patterns.

![TransGNN-AD Pipeline](images/BMVC_Flowchart_bg.png)

The pipeline consists of three main stages:
1. **Preprocessing** of diffusion MRI data, including tensor fitting and deterministic tractography
2. **Feature engineering** using normalized histograms of connectivity features from 9 neuroanatomically-significant ROIs
3. **Transductive learning** with Graph Neural Networks (GCN, GAT) that significantly outperform traditional classifiers in CN vs MCI and CN vs AD classification

## Repository Structure

```
TransGNN-AD/
├── Classification/               # Classification code for CN vs MCI and CN vs AD
│   ├── Inductive/              # Traditional ML methods (SVM, Random Forest, MLP, Transformer)
│   └── Transductive_graph/                    # Graph-based methods (GCN, GAT)
├── preprocessing_code/           # Preprocessing pipeline for diffusion MRI data
│   ├── Step1_*.py             # DICOM to NIfTI conversion 
│   ├── Step2_*.py             # Quantitative parameter extraction
│   ├── Step3_*.py             # DTI model and tractography 
│   ├── Step4_*.py             # Diffusion tensor generation
│   ├── Step5_*.py             # JHU atlas registration
│   ├── Step6_*.py             # ROI-specific parameter extraction
│   └── Step7_*.py             # Histogram feature generation
├── Dataset/                      # Sample dataset and instructions
│   └── example_subject/          # Example subject with raw data and preprocessed features
├── requirements.txt              # Package dependencies
└── README.md                     # Main documentation
```

## Installation

1. Clone this repository:
   ```bash
   git clone 
   cd TransGNN-AD
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv transGNN-AD_env
   source transGNN-AD_env/bin/activate  # On Windows, use: transGNN-AD_env\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data

We use diffusion MRI data from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database. Due to data access restrictions, users need to apply for access to the ADNI database directly.


An example subject with the expected data structure is provided in the `Dataset/example_subject/` directory.

## Usage

### Preprocessing Pipeline

The preprocessing pipeline consists of 7 steps, with notebooks provided for each step:

1. **Convert DICOM to NIfTI**:
   ```bash
   cd preprocessing_code
   jupyter notebook Step1_preprocessing_ADNI_AD.ipynb
   ```

2. Follow subsequent steps (2-7) in order to generate the histogram features for classification.

### Classification

Available options:
- `--model`: svm, rf, mlp, transformer, gcn, gat
- `--task`: CN_vs_MCI, CN_vs_AD

## Results

Our experimental results demonstrate remarkable improvements over conventional approaches:

- **CN vs MCI Classification**: GCN achieves 98.96% accuracy with high precision (99.13%) and recall (99.03%)
- **CN vs AD Classification**: GCN achieves 97.99% accuracy, while GAT maintains robust performance at 96.33%

Both significantly outperform traditional methods which average around 68-71% accuracy.

## Acknowledgments

Data used in the preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
