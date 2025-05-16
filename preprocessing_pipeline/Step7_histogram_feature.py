#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 7: Histogram Feature Extraction

This script generates histogram features from diffusion tensor parameters (FA, MD, AD) from specific brain regions.
It calculates normalized histograms for different regions of interest based on the JHU atlas.

Usage:
    python Step7_histogram_feature.py --mode 46 --sample

Requirements:
    - Python packages: numpy
"""

import os
import argparse
import numpy as np
from pathlib import Path


def generate_histograms(subject_id, mode, input_path, output_path, verbose=False):
    """
    Generate histogram features for a subject's diffusion tensor parameters.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    mode : int
        Number of diffusion directions (46, 54, or 55)
    input_path : str
        Path to the input data directory
    output_path : str
        Path to the output directory
    verbose : bool
        Whether to print detailed information
        
    Returns:
    --------
    bool
        Success status
    """
    try:
        # ROI labels (regions of interest from JHU atlas)
        # 5 - body of corpus callosum
        # 7 - fornix
        # 8 - corticospinal tract r
        # 16 - cerebral peduncle r
        # 18 - anterior limb of internal capsule r
        # 36, 37, 38, 39 - cingulum
        roi_labels = [5, 7, 8, 16, 18, 36, 37, 38, 39]
        
        # Set number of histogram bins
        no_of_bins = 20
        
        # Load the tensor parameter data
        input_folder = os.path.join(input_path, f"DTI_parameters/{mode}_diff_FA_MD_AD")
        
        # Load FA data
        fa_file = os.path.join(input_folder, f"FA_{subject_id}.npy")
        if not os.path.exists(fa_file):
            if verbose:
                print(f"FA file not found: {fa_file}")
            return False
            
        FA = np.load(fa_file, allow_pickle=True)
        
        # Load AD data
        ad_file = os.path.join(input_folder, f"AD_{subject_id}.npy")
        if not os.path.exists(ad_file):
            if verbose:
                print(f"AD file not found: {ad_file}")
            return False
            
        AD = np.load(ad_file, allow_pickle=True)
        
        # Load MD data
        md_file = os.path.join(input_folder, f"MD_{subject_id}.npy")
        if not os.path.exists(md_file):
            if verbose:
                print(f"MD file not found: {md_file}")
            return False
            
        MD = np.load(md_file, allow_pickle=True)
        
        if verbose:
            print(f"Loaded tensor parameters for subject {subject_id}")
            print(f"FA shape: {FA.shape}, AD shape: {AD.shape}, MD shape: {MD.shape}")
        
        # Process FA data
        bins_fa = np.linspace(0, 1, no_of_bins + 1)
        class_matrix = np.zeros((1, len(roi_labels) * no_of_bins))
        hist_array = np.zeros((len(roi_labels), no_of_bins))
        
        for j, fa_values in enumerate(FA):
            hist, _ = np.histogram(fa_values, bins=bins_fa)
            if len(fa_values) == 0:
                hist_array[j, :] = 0
            else:
                hist_array[j, :] = hist / len(fa_values)  # Normalize by the total number of FA values
                
        class_matrix[0, :] = hist_array.flatten()
        
        # Process AD data
        bins_ad = np.linspace(0, 20, no_of_bins + 1)  # Different range for AD values
        class_matrix_AD = np.zeros((1, len(roi_labels) * no_of_bins))
        hist_array_AD = np.zeros((len(roi_labels), no_of_bins))
        
        for j, ad_values in enumerate(AD):
            hist_AD, _ = np.histogram(ad_values, bins=bins_ad)
            if len(ad_values) == 0:
                hist_array_AD[j, :] = 0
            else:
                hist_array_AD[j, :] = hist_AD / len(ad_values)
        
        class_matrix_AD[0, :] = hist_array_AD.flatten()
        
        # Process MD data
        bins_md = np.linspace(0, 20, no_of_bins + 1)  # Different range for MD values
        class_matrix_MD = np.zeros((1, len(roi_labels) * no_of_bins))
        hist_array_MD = np.zeros((len(roi_labels), no_of_bins))
        
        for j, md_values in enumerate(MD):
            hist_MD, _ = np.histogram(md_values, bins=bins_md)
            if len(md_values) == 0:
                hist_array_MD[j, :] = 0
            else:
                hist_array_MD[j, :] = hist_MD / len(md_values)
        
        class_matrix_MD[0, :] = hist_array_MD.flatten()
        
        # Save histogram features
        output_folder = os.path.join(output_path, f"histogram_features/ClassMatrices_{mode}_diff")
        os.makedirs(output_folder, exist_ok=True)
        
        np.save(os.path.join(output_folder, f"class_matrix_FA_{subject_id}.npy"), class_matrix[0, :])
        np.save(os.path.join(output_folder, f"class_matrix_AD_{subject_id}.npy"), class_matrix_AD[0, :])
        np.save(os.path.join(output_folder, f"class_matrix_MD_{subject_id}.npy"), class_matrix_MD[0, :])
        
        if verbose:
            print(f"Saved histogram features for subject {subject_id}")
            print(f"Shape of class_matrix_FA: {class_matrix.shape}")
            print(f"Shape of class_matrix_AD: {class_matrix_AD.shape}")
            print(f"Shape of class_matrix_MD: {class_matrix_MD.shape}")
        
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
    dti_dir = subject_dir / "DTI_parameters" / "46_diff_FA_MD_AD"
    histogram_dir = subject_dir / "histogram_features" / "ClassMatrices_46_diff"
    
    dti_dir.mkdir(parents=True, exist_ok=True)
    histogram_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing sample subject: {subject_id}")
    print("This would extract histogram features from tensor parameters (FA, MD, AD).")
    print("Output would be saved to:", histogram_dir)
    
    print("\nFor each brain region, the script:")
    print("1. Calculates normalized histograms of tensor parameters")
    print("2. Creates feature vectors by flattening the histograms")
    print("3. Saves these features as numpy arrays for further analysis")
    
    print("\nTo run on real data, use the following command:")
    print("python Step7_histogram_feature.py --mode 46 --input_path /path/to/input --output_path /path/to/output --subjects subjects.txt")
    
    return True


def main():
    """Main function to parse arguments and execute the script"""
    parser = argparse.ArgumentParser(
        description='Generate histogram features from diffusion tensor parameters')
    parser.add_argument('--mode', type=int, choices=[46, 54, 55], 
                        help='Diffusion mode to process (46, 54, or 55)')
    parser.add_argument('--input_path', type=str, 
                        help='Path to the input base directory')
    parser.add_argument('--output_path', type=str, 
                        help='Path to the output base directory')
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
    
    # Validate arguments
    if not args.mode:
        print("Please specify diffusion mode with --mode (46, 54, or 55)")
        parser.print_help()
        return
    
    if not args.input_path or not args.output_path:
        print("Missing required paths. Please specify --input_path and --output_path")
        parser.print_help()
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
        # Try to find subjects from DTI parameters directory
        mode = args.mode
        dti_dir = os.path.join(args.input_path, f"DTI_parameters/{mode}_diff_FA_MD_AD")
        
        if os.path.exists(dti_dir):
            for filename in os.listdir(dti_dir):
                if filename.startswith('FA_') and filename.endswith('.npy'):
                    # Extract subject ID from filename: FA_subject_id.npy -> subject_id
                    subject_id = filename[3:-4]
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
        
        success = generate_histograms(
            subject_id=subject_id,
            mode=args.mode,
            input_path=args.input_path,
            output_path=args.output_path,
            verbose=args.verbose
        )
        
        if success:
            success_count += 1
    
    print(f"Successfully processed {success_count} of {len(subjects_list)} subjects for mode {args.mode}")


if __name__ == "__main__":
    main() 