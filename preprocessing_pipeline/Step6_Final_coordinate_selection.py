#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 6: Final Coordinate Selection

This script processes registered brain regions and extracts diffusion tensor parameters 
(FA, MD, AD) from specific regions of interest (ROIs) in the brain.
It uses the JHU atlas registered to each subject and KD-Tree for spatial lookup.

Usage:
    python Step6_Final_coordinate_selection.py --mode 46 --sample

Requirements:
    - Python packages: dipy, nibabel, numpy, scikit-learn
"""

import os
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path

from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import TensorModel, fractional_anisotropy, mean_diffusivity, axial_diffusivity
from sklearn.neighbors import KDTree


def load_nifti_data(file_path, return_img=False):
    """
    Load NIFTI file data.
    
    Parameters:
    -----------
    file_path : str
        Path to the NIFTI file
    return_img : bool
        Whether to return the NIFTI image object
        
    Returns:
    --------
    tuple
        (data, affine) or (data, affine, img)
    """
    nifti_img = nib.load(file_path)
    data = nifti_img.get_fdata()
    affine = nifti_img.affine
    if return_img:
        return data, affine, nifti_img
    return data, affine


def process_subject(subject_id, mode, input_base_path, output_base_path, jhu_template_path, 
                   roi_labels, reference_subject, verbose=False):
    """
    Process a single subject to extract diffusion tensor parameters from ROIs.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    mode : int
        Number of diffusion directions (46, 54, or 55)
    input_base_path : str
        Path to the input base directory
    output_base_path : str
        Path to the output base directory
    jhu_template_path : str
        Path to the JHU template
    roi_labels : list
        List of ROI label IDs to process
    reference_subject : dict
        Dictionary with reference subject information for bval/bvec
    verbose : bool
        Whether to print detailed information
        
    Returns:
    --------
    bool
        Success status
    """
    try:
        # Setup paths
        subject_folder_path = os.path.join(input_base_path, f"registered_data")
        registered_label_file_path = os.path.join(subject_folder_path, 
                                               f"{mode}_diff_registered_{subject_id}.nii.gz")
        
        # Check if registered label file exists
        if not os.path.exists(registered_label_file_path):
            print(f"Registered label file not found: {registered_label_file_path}")
            return False
        
        # Load registered labels
        registered_label_img = nib.load(registered_label_file_path)
        registered_labels = registered_label_img.get_fdata()
        
        # Find coordinates corresponding to the registered labels
        label_coordinates = []
        
        for label in roi_labels:
            label_coordinates_label = np.array(np.where(registered_labels == label)).T
            label_coordinates.append(label_coordinates_label)
            
        # Load and prepare coordinates
        coordinates_folder_path = os.path.join(output_base_path, f"coordinates/{mode}_diff_coordinates")
        coordinates_file_path = os.path.join(coordinates_folder_path, f"coordinates_{subject_id}.txt")
        
        if not os.path.exists(coordinates_file_path):
            print(f"Coordinates file not found: {coordinates_file_path}")
            return False
        
        valid_coordinates = np.loadtxt(coordinates_file_path, delimiter=',')
        
        # KDTree for quick spatial lookup
        tree = KDTree(np.round(valid_coordinates), leaf_size=2, metric='l2')
        
        # Select only those coordinates from the registered labels
        selected_coordinates = []
        for label_coord in label_coordinates:
            if label_coord.shape[0] != 0:
                dist, indices = tree.query(label_coord, k=1)
                # Filter by distance threshold
                indices = indices[dist <= 4]  
                valid = valid_coordinates[indices]
                selected_coordinates.append(valid)
                if verbose:
                    print(f"Found {len(selected_coordinates)} ROIs processed")
        
        # Load diffusion tensor data
        ordered_folder_path = os.path.join(input_base_path, f"ordered_4d/{mode}_diff_ordered")
        nii_gz_file = os.path.join(ordered_folder_path, f"4d_img_DTI_{subject_id}.nii.gz")
        
        # Check if 4D data file exists
        if not os.path.exists(nii_gz_file):
            print(f"4D data file not found: {nii_gz_file}")
            return False
        
        data, affine, hardi_img = load_nifti_data(nii_gz_file, return_img=True)
        
        # Load reference b-values and b-vectors
        ref_subject_id = reference_subject.get(str(mode))
        bval_file = os.path.join(input_base_path, f"subjects/{ref_subject_id}/{ref_subject_id}_bval.npy")
        bvec_file = os.path.join(input_base_path, f"subjects/{ref_subject_id}/{ref_subject_id}_bvec.npy")
        
        # Check if bval and bvec files exist
        if not os.path.exists(bval_file) or not os.path.exists(bvec_file):
            print(f"bval/bvec files not found: {bval_file}, {bvec_file}")
            return False
            
        common_bval_file = np.load(bval_file)
        common_bvec_file = np.load(bvec_file)
        
        gtab = gradient_table(common_bval_file, bvecs=common_bvec_file)
        
        # Create a mask for the data
        maskdata, mask = median_otsu(data, vol_idx=range(10, min(46, data.shape[-1])), 
                                   median_radius=3, numpass=1, autocrop=False, dilate=2)
        
        # Tensor model fitting
        tenmodel = TensorModel(gtab)
        tenfit = tenmodel.fit(maskdata)
        
        # Calculate diffusion parameters for each ROI
        FA = []
        MD = []
        AD = []
        
        for s in selected_coordinates:
            selected_coordinates_indices = s.astype(float)
            selected_coordinates_indices = tuple(selected_coordinates_indices.T.astype(int).tolist())
            
            try:
                selected_evals = tenfit.evals[selected_coordinates_indices]
                
                # Calculate tensor-derived parameters
                FA.append(fractional_anisotropy(selected_evals))
                MD.append(mean_diffusivity(selected_evals))
                AD.append(axial_diffusivity(selected_evals))
            except Exception as e:
                print(f"Error calculating tensor values: {e}")
                # Append empty arrays if calculation fails
                FA.append(np.array([]))
                MD.append(np.array([]))
                AD.append(np.array([]))
        
        if verbose:
            print(f"Type of FA: {type(FA)}, Length: {len(FA)}")
        
        # Save as object-compatible numpy arrays
        FA = np.array(FA, dtype=object)
        MD = np.array(MD, dtype=object)
        AD = np.array(AD, dtype=object)
        
        # Create output directory
        output_folder = os.path.join(output_base_path, f"DTI_parameters/{mode}_diff_FA_MD_AD")
        os.makedirs(output_folder, exist_ok=True)
        
        # Save tensor-derived parameters
        np.save(os.path.join(output_folder, f"FA_{subject_id}.npy"), FA)
        np.save(os.path.join(output_folder, f"MD_{subject_id}.npy"), MD)
        np.save(os.path.join(output_folder, f"AD_{subject_id}.npy"), AD)
        np.savez(os.path.join(output_folder, f"{subject_id}_FA_MD_AD.npz"), FA=FA, MD=MD, AD=AD)
        
        if verbose:
            print(f"Saved tensor parameters for subject {subject_id}")
            
        return True
    
    except Exception as e:
        print(f"Error processing subject {subject_id}: {e}")
        return False


def process_sample_subject():
    """Process a sample subject for demonstration"""
    # Define paths for the sample subject
    base_dir = Path("sample_data")
    subject_id = "sample_subject"
    
    # Create directory structure
    base_dir.mkdir(parents=True, exist_ok=True)
    
    subject_dir = base_dir / "output"
    coordinates_dir = subject_dir / "coordinates" / "46_diff_coordinates" 
    output_dir = subject_dir / "DTI_parameters" / "46_diff_FA_MD_AD"
    
    coordinates_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define ROI labels (example subset of JHU atlas regions)
    roi_labels = [5, 7, 8, 16, 18, 36, 37, 38, 39]
    
    print(f"Processing sample subject: {subject_id}")
    print("Sample processing would extract tensor parameters from ROIs.")
    print("Output would be saved to:", output_dir)
    
    print("\nTo run on real data, use the following command:")
    print("python Step6_Final_coordinate_selection.py --mode 46 --input_path /path/to/input --output_path /path/to/output --subjects subjects.txt")
    
    return True


def main():
    """Main function to parse arguments and execute the script"""
    parser = argparse.ArgumentParser(
        description='Extract diffusion tensor parameters from regions of interest')
    parser.add_argument('--mode', type=int, choices=[46, 54, 55], 
                        help='Diffusion mode to process (46, 54, or 55)')
    parser.add_argument('--input_path', type=str, 
                        help='Path to the input base directory')
    parser.add_argument('--output_path', type=str, 
                        help='Path to the output base directory')
    parser.add_argument('--jhu_template', type=str, 
                        help='Path to the JHU template file')
    parser.add_argument('--subjects_file', type=str, 
                        help='Path to a text file containing subject IDs')
    parser.add_argument('--subject', type=str, 
                        help='Process a single subject ID')
    parser.add_argument('--sample', action='store_true', 
                        help='Process sample subject')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Process sample subject if requested
    if args.sample:
        process_sample_subject()
        return
    
    # Define reference subjects for each mode
    reference_subjects = {
        '46': "016_S_4009",
        '54': "027_S_6648",
        '55': "011_S_4827"
    }
    
    # Define ROI labels (regions of interest from JHU atlas)
    roi_labels = [5, 7, 8, 16, 18, 36, 37, 38, 39]
    
    # Validate arguments
    if not args.mode:
        print("Please specify diffusion mode with --mode (46, 54, or 55)")
        parser.print_help()
        return
    
    if not args.input_path or not args.output_path:
        print("Missing required paths. Please specify --input_path and --output_path")
        parser.print_help()
        return
    
    # Set JHU template path
    jhu_template_path = args.jhu_template
    if not jhu_template_path:
        jhu_template_path = os.path.join(args.input_path, "templates", "JHU-ICBM-labels-1mm.nii.gz")
    
    if not os.path.exists(jhu_template_path):
        print(f"JHU template not found: {jhu_template_path}")
        print("Please specify --jhu_template with the correct path")
        return
    
    # Get list of subjects
    subjects_list = []
    
    if args.subject:
        # Process a single subject
        subjects_list = [args.subject]
    elif args.subjects_file:
        # Load subjects from file
        try:
            with open(args.subjects_file, 'r') as f:
                subjects_list = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading subjects file: {e}")
            return
    else:
        # Try to find subjects from coordinates directory
        mode = args.mode
        coordinates_dir = os.path.join(args.output_path, f"coordinates/{mode}_diff_coordinates")
        
        if os.path.exists(coordinates_dir):
            for filename in os.listdir(coordinates_dir):
                if filename.startswith('coordinates_') and filename.endswith('.txt'):
                    subject_id = filename[12:-4]  # Extract subject ID from filename
                    subjects_list.append(subject_id)
        
        if not subjects_list:
            print("No subjects found. Please specify --subjects_file or --subject")
            return
    
    print(f"Processing {len(subjects_list)} subjects for mode {args.mode}")
    
    # Process each subject
    success_count = 0
    for subject_id in subjects_list:
        if args.verbose:
            print(f"\nProcessing subject: {subject_id}")
        
        success = process_subject(
            subject_id=subject_id,
            mode=args.mode,
            input_base_path=args.input_path,
            output_base_path=args.output_path,
            jhu_template_path=jhu_template_path,
            roi_labels=roi_labels,
            reference_subject=reference_subjects,
            verbose=args.verbose
        )
        
        if success:
            success_count += 1
    
    print(f"Successfully processed {success_count} of {len(subjects_list)} subjects for mode {args.mode}")


if __name__ == "__main__":
    main() 