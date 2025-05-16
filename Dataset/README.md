# ADNI Dataset for TransGNN-AD

This folder contains preprocessed data from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database. If you're looking to download the original data and reproduce our preprocessing steps, follow the instructions below.

## Downloading Data from ADNI

The ADNI dataset is not publicly available and requires registration and approval. Here's how to obtain access:

1. Visit the ADNI website: [https://adni.loni.usc.edu/](https://adni.loni.usc.edu/)

2. Click on "Apply for Data Access" or navigate to "Data & Samples" → "Access Data"

3. Complete the registration form:
   - You'll need to provide your personal information
   - Specify your institutional affiliation
   - Describe your research project and intended use of the data
   - Accept the Data Use Agreement

4. Wait for approval (typically takes 1-5 business days)

5. Once approved, log in to the ADNI data portal

## Downloading DTI Data

For this project, we specifically used DTI (Diffusion Tensor Imaging) data:

1. After logging in, go to "Advanced Search"

2. Under "Modality", select "DTI"

3. Filter subjects by diagnosis groups:
   - Alzheimer's Disease (AD)
   - Mild Cognitive Impairment (MCI)
   - Cognitively Normal (CN)

4. Select the specific subjects and timepoints you need

5. Add the selected data to your download cart

6. Follow the download instructions provided by the ADNI website

## Preprocessing Pipeline

To process the raw ADNI DTI data, use the scripts in the `preprocessing_pipeline` folder in the following order:

1. `Step1_DICOM_to_NIfTI_conversion.py` - Converts DICOM files to NIfTI format
2. `Step2_Finding_Quantitative_Parameters.py` - Extracts DTI metrics like FA (Fractional Anisotropy) and MD (Mean Diffusivity)
3. `Step3_DTIODFmodel_final_ADNI_full_pipeline.py` - Processes DTI data using deterministic tractography models
4. `Step4_Generate_Diffusion_Tensor.py` - Creates diffusion tensor models from processed DTI data
5. `Step5_JHU_label_registration.py` - Registers brain regions using JHU brain atlas for anatomical reference
6. `Step6_Final_coordinate_selection.py` - Selects specific fiber tract coordinates for connectivity analysis
7. `Step7_histogram_feature.py` - Generates histogram-based features for the TransGNN-AD model

Each script has specific requirements and dependencies which are listed in the project's main `requirements.txt` file. For detailed usage of each script, please refer to the comments and documentation within each file.

## Dataset Structure

After preprocessing, the Dataset folder is organized as follows:

```
Dataset/
├── example_subject/              # Example of processed subject data
│   └── raw/                      # Raw DTI data in NIfTI format 
│       └── 4d_img_DTI_*.nii.gz
├── histogram_features_all_subjects/  # Processed features for model input
│   ├── AD/                       # Alzheimer's Disease subjects
│   │   └── ClassMatrices_*/      # Different class matrix configurations
│   ├── CN/                       # Cognitively Normal subjects
│   │   └── ClassMatrices_*/
│   └── MCI/                      # Mild Cognitive Impairment subjects
│       └── ClassMatrices_*/
└── README.md                     # This file
```

Each `ClassMatrices_*` folder contains the processed brain connectivity matrices in `.npy` format, organized by DTI metric type (AD, FA, MD) and subject ID.

## Reference

When using this data, please cite both the ADNI database and our work:

```
Data collection and sharing for this project was funded by the Alzheimer's Disease Neuroimaging Initiative (ADNI) (National Institutes of Health Grant U01 AG024904) and DOD ADNI (Department of Defense award number W81XWH-12-2-0012).
```

For questions regarding the ADNI database, please visit [http://adni.loni.usc.edu/about/contact/](http://adni.loni.usc.edu/about/contact/)