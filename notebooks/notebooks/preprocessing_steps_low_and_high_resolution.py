#!/usr/bin/env python3
import os, time
import numpy as np
import nibabel as nib
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from skimage.filters import gaussian
import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore")

# === DIRECTORIES ===
# Set your local directories here before running the script.
# Example:
# linear_dir = '/path/to/downsampled_linear_matlab'
# output_dir = '/path/to/preprocessed_registered_output'
linear_dir = '/path/to/downsampled_linear_matlab'  # CHANGE THIS to your local directory
output_dir = '/path/to/preprocessed_registered_output'  # CHANGE THIS to your local directory

transforms_dir = os.path.join(output_dir, 'transforms')
optimizer_log_dir = os.path.join(output_dir, 'optimizer_logs')
os.makedirs(transforms_dir, exist_ok=True)
os.makedirs(optimizer_log_dir, exist_ok=True)

# === HELPER FUNCTIONS ===
def load_volume(path):
    """Load a NIfTI volume and return numpy array."""
    return nib.load(path).get_fdata()

def normalize(img):
    """Normalize intensities to [0,1]."""
    return (img - img.min()) / (img.max() - img.min())

# === METRICS NOTE ===
# Evaluation metrics are computed in the dedicated script:
# notebooks/metrics_throughout_pipeline.py
# These metrics have been consistently used throughout the entire thesis.

# === PREPROCESSING AND REGISTRATION PIPELINE ===
# This preprocessing pipeline is the same for both low- and high-resolution data.
# The only difference is in directory paths and the number of parallel processes used.
def process(pid):
    pid_str = f'{pid:02d}'
    try:
        start_time = time.time()

        # Load and normalize perfusion volume
        perf = normalize(load_volume(os.path.join(linear_dir, f'perf_{pid_str}_TRJ.nii')))
        # Load DTI volume
        dti = load_volume(os.path.join(linear_dir, f'dti_{pid_str}_TRJ.nii'))

        # Gaussian smoothing of DTI
        # sigma=0.3 was chosen for optimal smoothing on our data; can be adjusted per dataset
        dti = gaussian(dti, sigma=0.3, preserve_range=True)
        dti = normalize(dti)

        # Convert to SimpleITK images
        fixed = sitk.Cast(sitk.GetImageFromArray(perf.astype(np.float32)), sitk.sitkFloat32)
        moving = sitk.Cast(sitk.GetImageFromArray(dti.astype(np.float32)), sitk.sitkFloat32)

        # Rigid registration parameters:
        # bins=128 for Mattes Mutual Information metric (adjustable)
        # learning rate, iterations, shrink factors and smoothing sigmas
        rig = sitk.ImageRegistrationMethod()
        rig.SetMetricAsMattesMutualInformation(128)
        rig.SetMetricSamplingStrategy(rig.NONE)
        rig.SetInterpolator(sitk.sitkLinear)
        rig.SetOptimizerAsRegularStepGradientDescent(learningRate=0.15, minStep=1e-6, numberOfIterations=50)
        rig.SetShrinkFactorsPerLevel([2,1])
        rig.SetSmoothingSigmasPerLevel([1,0])
        rig.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        rig.SetInitialTransform(sitk.Euler3DTransform(), inPlace=True)
        tf_rigid = rig.Execute(fixed, moving)
        sitk.WriteTransform(tf_rigid, os.path.join(transforms_dir, f'transform_rigid_TRJ_{pid_str}.tfm'))

        # Affine registration parameters (initialized from rigid):
        # same MI bins, slightly higher learning rate
        afi = sitk.ImageRegistrationMethod()
        afi.SetMetricAsMattesMutualInformation(128)
        afi.SetMetricSamplingStrategy(afi.NONE)
        afi.SetInterpolator(sitk.sitkLinear)
        afi.SetOptimizerAsRegularStepGradientDescent(learningRate=0.20, minStep=1e-6, numberOfIterations=50)
        afi.SetShrinkFactorsPerLevel([2,1])
        afi.SetSmoothingSigmasPerLevel([1,0])
        afi.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        afi.SetInitialTransform(tf_rigid, inPlace=True)
        tf_affine = afi.Execute(fixed, moving)
        sitk.WriteTransform(tf_affine, os.path.join(transforms_dir, f'transform_affine_TRJ_{pid_str}.tfm'))

        # Warp moving image with affine transform and save
        warped = sitk.Resample(moving, fixed, tf_affine, sitk.sitkLinear, 0.0, moving.GetPixelID())
        arr = sitk.GetArrayFromImage(warped)
        output_path = os.path.join(output_dir, f'dti_{pid_str}_reg_final.nii')
        nib.save(nib.Nifti1Image(arr, np.eye(4)), output_path)

        elapsed_time = time.time() - start_time
        print(f"[TRJ_{pid_str}] Registration completed in {elapsed_time:.2f} seconds.")
        return {
            'Patient': f'TRJ_{pid_str}',
            'ProcessingTime_sec': elapsed_time,
            'OutputPath': output_path
        }

    except Exception as e:
        print(f"Error processing TRJ_{pid_str}: {e}")
        return None

if __name__ == '__main__':
    # We used 60 patients total, excluding patients 9 and 22 who were missing data.
    # Adjust this list according to your dataset.
    patients = [i for i in range(1, 61) if i not in (9, 22)]

    # Multiprocessing Pool: In low resolution (e.g. in Colab), I run ~30 patients in parallel
    # to accelerate computation time.
    with mp.Pool(30) as pool:
        results = list(tqdm(pool.imap(process, patients), total=len(patients)))

    results = [r for r in results if r is not None]
    df = pd.DataFrame(results).set_index('Patient')

    # Save processing times and output paths CSV
    csv_path = os.path.join(output_dir, 'preprocessing_times_and_outputs.csv')
    df.to_csv(csv_path)
    print(f"\nProcessing times and outputs saved to {csv_path}")

