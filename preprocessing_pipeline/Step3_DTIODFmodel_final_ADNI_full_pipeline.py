#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 3: Generate DTI ODF Model - ADNI Full Pipeline

This script creates a DTI ODF model from preprocessed diffusion MRI data.
It fits the diffusion tensor model, calculates fractional anisotropy, and performs tractography.

Usage:
    python Step3_DTIODFmodel_final_ADNI_full_pipeline.py --subject_path /path/to/subject --output_path /path/to/output

Requirements:
    - Python packages: dipy, nibabel, numpy, matplotlib
    - Dependencies: fury, vtk
"""

import os
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.reconst.dti import color_fa
from dipy.direction import peaks_from_model
from dipy.segment.mask import median_otsu
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.data import get_sphere
import argparse
from pathlib import Path


def process_subject(subject_path, subject_id, output_path=None):
    """
    Process a subject to create a DTI ODF model and perform tractography.
    
    Parameters:
    -----------
    subject_path : str
        Path to the subject's directory
    subject_id : str
        Subject identifier
    output_path : str, optional
        Output directory (if different from subject_path)
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    if output_path is None:
        output_path = subject_path
    
    # Check if tractogram already exists
    trk_file = os.path.join(output_path, f'tractogram_deterministic_{subject_id}_DTI.trk')
    if os.path.exists(trk_file):
        print(f"Skipping {subject_id}, tractogram already exists.")
        return True
    
    try:
        # Find the nifti file
        nifti_files = [f for f in os.listdir(subject_path) if f.endswith('.nii.gz')]
        if not nifti_files:
            raise FileNotFoundError(f"No NIFTI file found for subject {subject_id}")
        
        nii_file = os.path.join(subject_path, nifti_files[0])
        print(f"Loading {nii_file}")
        
        # Load the data
        data, affine, hardi_img = load_nifti(nii_file, return_img=True)
        print('Data shape:', data.shape)
        
        # Check if the data has the expected shape
        if data.shape[-1] not in [46, 54, 55]:
            print(f"Skipping {subject_id}, shape {data.shape} does not match criteria.")
            return False
        
        # Load bvals and bvecs
        if os.path.exists(os.path.join(subject_path, f"{subject_id}_bval.npy")):
            bvals = np.load(os.path.join(subject_path, f"{subject_id}_bval.npy"))
            bvecs = np.load(os.path.join(subject_path, f"{subject_id}_bvec.npy"))
        else:
            # Try to find any bval/bvec file
            bval_files = [f for f in os.listdir(subject_path) if f.endswith('_bval.npy') or f.endswith('.bval')]
            bvec_files = [f for f in os.listdir(subject_path) if f.endswith('_bvec.npy') or f.endswith('.bvec')]
            
            if not bval_files or not bvec_files:
                raise FileNotFoundError(f"No bval/bvec files found for subject {subject_id}")
            
            if bval_files[0].endswith('.npy'):
                bvals = np.load(os.path.join(subject_path, bval_files[0]))
            else:
                bvals = np.loadtxt(os.path.join(subject_path, bval_files[0]))
                
            if bvec_files[0].endswith('.npy'):
                bvecs = np.load(os.path.join(subject_path, bvec_files[0]))
            else:
                bvecs = np.loadtxt(os.path.join(subject_path, bvec_files[0]))
        
        # Create gradient table
        gtab = gradient_table(bvals=bvals, bvecs=bvecs)
        
        # Masking
        maskdata, mask = median_otsu(data, 
                                    vol_idx=range(10, min(46, data.shape[-1])), 
                                    median_radius=3, 
                                    numpass=1, 
                                    autocrop=False, 
                                    dilate=2)
        print('Maskdata shape:', maskdata.shape)
        
        # Tensor fitting
        tenmodel = TensorModel(gtab)
        tenfit = tenmodel.fit(maskdata)
        
        # Get sphere
        sphere = get_sphere(name='repulsion724')
        
        # Calculate FA
        FA = fractional_anisotropy(tenfit.evals)
        FA = np.clip(FA, 0, 1)
        RGB = color_fa(FA, tenfit.evecs)
        
        # Get peaks
        maskdata1 = maskdata[:, :, :, 0]
        dti_peaks = peaks_from_model(tenmodel, data, sphere=sphere,
                                    relative_peak_threshold=.8,
                                    min_separation_angle=45,
                                    normalize_peaks=False,
                                    mask=maskdata1,
                                    sh_order_max=8,
                                    npeaks=1)
        
        # Tractography
        FA_mask = FA > 0.2
        seed_mask = FA > 0.8
        stopping_criterion = ThresholdStoppingCriterion(FA, .25)
        seeds = utils.seeds_from_mask(seed_mask, density=2, affine=affine)
        
        streamlines_generator = LocalTracking(dti_peaks, stopping_criterion, seeds,
                                            affine=affine, step_size=.5)
        streamlines = Streamlines(streamlines_generator)
        
        # Save tractogram
        sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
        save_trk(sft, trk_file)
        
        print(f'Processing complete for {subject_id}')
        return True
        
    except Exception as e:
        print(f"Error processing {subject_id}: {str(e)}")
        return False


def process_all_subjects(base_path, output_base=None):
    """
    Process all subjects in the dataset.
    
    Parameters:
    -----------
    base_path : str
        Path to the base directory containing subject folders
    output_base : str, optional
        Base output directory (if different from base_path)
    """
    # Get list of subject directories
    subjects = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    if not subjects:
        print(f"No subject directories found in {base_path}")
        return
    
    processed_count = 0
    for subject_id in subjects:
        subject_path = os.path.join(base_path, subject_id)
        
        # Determine output directory
        if output_base:
            subject_output_dir = os.path.join(output_base, subject_id)
            os.makedirs(subject_output_dir, exist_ok=True)
        else:
            subject_output_dir = subject_path
        
        success = process_subject(subject_path, subject_id, subject_output_dir)
        if success:
            processed_count += 1
    
    print(f"Completed processing {processed_count} out of {len(subjects)} subjects.")


def process_sample_subject():
    """Process the sample subject for demonstration"""
    from pathlib import Path
    
    # Define paths for the sample subject
    base_dir = Path("sample_data")
    subject_id = "sample_subject"
    subject_dir = base_dir / subject_id
    
    # Ensure sample data directory exists
    subject_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if sample data exists
    if not list(subject_dir.glob("*.nii.gz")):
        print("Sample data not found. Please download the sample data first.")
        return False
    
    print(f"Processing sample subject: {subject_id}")
    return process_subject(str(subject_dir), subject_id)


def main():
    """Main function to parse arguments and execute the script"""
    parser = argparse.ArgumentParser(description='Create DTI ODF model and perform tractography')
    parser.add_argument('--subject_path', type=str, help='Path to subject directory')
    parser.add_argument('--subject_id', type=str, help='Subject ID')
    parser.add_argument('--output_path', type=str, help='Output directory')
    parser.add_argument('--all_subjects', action='store_true', help='Process all subjects in directory')
    parser.add_argument('--sample', action='store_true', help='Process sample subject')
    
    args = parser.parse_args()
    
    if args.sample:
        process_sample_subject()
    elif args.all_subjects and args.subject_path:
        process_all_subjects(args.subject_path, args.output_path)
    elif args.subject_path and args.subject_id:
        process_subject(args.subject_path, args.subject_id, args.output_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 