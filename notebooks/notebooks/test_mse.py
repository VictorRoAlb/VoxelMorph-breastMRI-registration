#!/usr/bin/env python

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import voxelmorph as vxm
import nibabel as nib
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from scipy.stats import pearsonr
from skimage.filters import threshold_otsu
from medpy.metric.binary import dc, hd95
import warnings


# === CONFIGURATION ===
# Base directory for the project (adjust according to your environment)
BASE_DIR         = '/content/drive/MyDrive/TFG_Victor'

# Directory where trained MSE model weights are stored
WEIGHTS_DIR      = os.path.join(BASE_DIR, 'MSE - pesos guardados')

# Filename of the final trained weights for MSE model
MODEL_FILENAME   = 'mse_perf_dti_final_weights.h5'

# Full path to trained model weights
MODEL_PATH       = os.path.join(WEIGHTS_DIR, MODEL_FILENAME)

# Test data split list file (train-test partition consistent with training)
TEST_LIST        = os.path.join(BASE_DIR, 'particiones', 'test_list.txt')

# Output directories for test results
OUTPUT_DIR       = os.path.join(BASE_DIR, 'test_results_MSE')
VOL_DIR          = os.path.join(OUTPUT_DIR, 'test_volumes_corregistrados_MSE')  # Registered volumes output folder
FLOW_DIR         = os.path.join(OUTPUT_DIR, 'campos_deformacion')              # Deformation fields output folder

# Create output directories if they do not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VOL_DIR, exist_ok=True)
os.makedirs(FLOW_DIR, exist_ok=True)

# === METRICS NOTE ===
# Evaluation metrics computed here are consistent with those defined in
# notebooks/metrics_throughout_pipeline.py, ensuring uniformity across the pipeline.
# Metrics calculated include: SSIM, NCC, MSE, Dice, Hausdorff95, and Mattes Mutual Information.

def mutual_info_mattes(vol1, vol2, bins=128):
    img1 = sitk.GetImageFromArray(vol1.astype(np.float32))
    img2 = sitk.GetImageFromArray(vol2.astype(np.float32))
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(bins)
    reg.SetMetricSamplingStrategy(reg.NONE)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-6, 1)
    reg.SetInitialTransform(sitk.TranslationTransform(img1.GetDimension()))
    reg.Execute(img1, img2)
    return -reg.GetMetricValue()

def calc_all_metrics(moved, fixed):
    ssim_vals, ncc_vals = [], []
    for z in range(fixed.shape[2]):
        f0 = fixed[:,:,z]
        m0 = moved[:,:,z]
        if f0.std() > 0 and m0.std() > 0:
            ssim_vals.append(ssim(f0, m0, data_range=1))
            ncc_vals.append(np.corrcoef(f0.ravel(), m0.ravel())[0,1])

    flat_f = fixed.ravel()
    flat_m = moved.ravel()
    mse_val = mean_squared_error(flat_f, flat_m)
    # Pearson correlation removed as per project requirements

    t_f = threshold_otsu(fixed)
    t_m = threshold_otsu(moved)
    b_f = fixed > t_f
    b_m = moved > t_m

    dice_val = dc(b_f, b_m)
    haus95_val = hd95(b_f, b_m)
    mi_val = mutual_info_mattes(fixed, moved, bins=128)

    return {
        'SSIM': np.mean(ssim_vals) if ssim_vals else np.nan,
        'NCC': np.mean(ncc_vals) if ncc_vals else np.nan,
        'MSE': mse_val,
        'Hausdorff95': haus95_val,
        'Dice': dice_val,
        'MattesMutualInfo': mi_val
    }

# === MODEL LOADING AND DEVICE SETUP ===
# Device selection follows original VoxelMorph strategy:
# use GPU if available, else CPU.
device, _ = vxm.tf.utils.setup_device(None)
with tf.device(device):
    # Load pretrained model weights
    vm_model  = vxm.networks.VxmDense.load(MODEL_PATH, input_model=None)
    reg_model = vm_model.get_registration_model()

    # Input shape extraction
    inshape   = reg_model.inputs[0].shape[1:-1]

    # Transformation layer uses linear interpolation
    # (original script uses nearest neighbor, which is better suited for segmentation maps)
    transform = vxm.networks.Transform(inshape, interp_method='linear')

# === READ TEST PAIRS ===
with open(TEST_LIST, 'r') as f:
    raw_pairs = [line.strip().split(',') for line in f if line.strip()]

# Order inversion: perfusion fixed image, DTI moving image
pairs = [(perf_path, dti_path) for dti_path, perf_path in raw_pairs]

results = []

for i, (fix_path, mov_path) in enumerate(pairs, 1):
    # Load volumes with batch and feature axis for TF compatibility
    fixed  = vxm.py.utils.load_volfile(fix_path, np_var='vol', add_batch_axis=True, add_feat_axis=True)
    moving = vxm.py.utils.load_volfile(mov_path, np_var='vol', add_batch_axis=True, add_feat_axis=True)

    # Predict deformation and measure time
    start = time.time()
    flow  = reg_model.predict([moving, fixed])
    t_reg = time.time() - start

    # Warp moving volume
    moved     = transform.predict([moving, flow])[0,...,0]
    fixed_img = fixed[0,...,0]

    # Calculate metrics consistent with project evaluation
    mets = calc_all_metrics(moved, fixed_img)
    mets.update({
        'Patient': os.path.basename(fix_path),
        'Time_s': round(t_reg, 3)
    })
    results.append(mets)

    print(f"[{i}/{len(pairs)}] {mets['Patient']} — "
          f"Time: {mets['Time_s']}s | MSE: {mets['MSE']:.4f} | "
          f"SSIM: {mets['SSIM']:.4f} | NCC: {mets['NCC']:.4f} | "
          f"Hausdorff95: {mets['Hausdorff95']:.1f} | "
          f"Dice: {mets['Dice']:.4f} | MI: {mets['MattesMutualInfo']:.4f}")

    # Save registered volume using original affine matrix (added feature)
    orig = nib.load(mov_path)
    out_nii = nib.Nifti1Image(moved.astype(np.float32), orig.affine)
    nib.save(out_nii, os.path.join(VOL_DIR, f"reg_{os.path.basename(mov_path)}"))

    # Save deformation fields for further analysis (added feature)
    flow_out = os.path.join(FLOW_DIR, f"flow_{os.path.basename(mov_path).replace('.nii','')}.npz")
    np.savez_compressed(flow_out, flow=flow[0])

# Save detailed metrics CSV file
df = pd.DataFrame(results)
csv_full = os.path.join(OUTPUT_DIR, 'test_metrics_detalladas_MSE.csv')
df.to_csv(csv_full, index=False)

# Save summary CSV with mean and std
numeric = df.drop(columns=['Patient', 'Time_s'])
summary = numeric.agg(['mean', 'std']).T
csv_summary = os.path.join(OUTPUT_DIR, 'test_metrics_MSE.csv')
summary.to_csv(csv_summary, index=True, header=['mean', 'std'])

print("\n Detailed metrics saved at:", csv_full)
print("Summary metrics saved at:", csv_summary)
print(" Registered volumes saved at:", VOL_DIR)
print(" Deformation fields saved at:", FLOW_DIR)
