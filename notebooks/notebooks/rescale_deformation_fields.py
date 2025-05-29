#!/usr/bin/env python

import os
import time
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf
import SimpleITK as sitk
from scipy.ndimage import zoom, map_coordinates
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from scipy.stats import pearsonr
from medpy.metric.binary import dc, hd95
from skimage.filters import threshold_otsu
import warnings

# === NOTE ===
# In this script, only the deformation fields from the NCC model are rescaled to high resolution (HR).
# This decision is based on test results, where the NCC-trained model showed superior performance
# compared to the MSE model. Therefore, only NCC volumes are upsampled and evaluated in HR space.

# === CONFIGURATION ===
# Set input/output paths and directory structure
BASE_DIR     = "/content/drive/MyDrive/TFG_Victor"
FLOW_DIR     = os.path.join(BASE_DIR, "test_results_NCC", "campos_deformacion_NCC")  # Deformation fields from NCC
HR_DIR       = os.path.join(BASE_DIR, "Pacientes_en_Alta_Resolucion_tras_preprocesado")  # High-resolution input volumes
OUT_DIR      = os.path.join(BASE_DIR, "Resultados_TEST_reescalado_NCC")  # Main output directory
VOLS_DIR     = os.path.join(OUT_DIR, "volumenes_registrados_HR")  # Folder for warped HR DTI volumes
FLOWS_DIR    = os.path.join(OUT_DIR, "campos_reescalados_HR")  # Folder for rescaled deformation fields
os.makedirs(VOLS_DIR, exist_ok=True)
os.makedirs(FLOWS_DIR, exist_ok=True)

# Test patients selected from test_list.txt (NCC-trained model only)
pacientes_test = ['01', '06', '37', '15', '47', '55', '40', '28', '49']

# Silence unnecessary warnings and TensorFlow logs
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# === METRICS NOTE ===
# Metrics are aligned with those used throughout the full pipeline
# (see metrics_throughout_pipeline.py for consistency)
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

def calc_metrics(fixed, moved):
    # Compute full set of evaluation metrics slice-by-slice and globally
    ssim_vals, ncc_vals = [], []
    for z in range(fixed.shape[2]):
        f, m = fixed[:,:,z], moved[:,:,z]
        if f.std() > 0 and m.std() > 0:
            ssim_vals.append(ssim(f, m, data_range=1))
            ncc_vals.append(np.corrcoef(f.ravel(), m.ravel())[0,1])
    flat_f = fixed.ravel()
    flat_m = moved.ravel()
    t1 = threshold_otsu(fixed)
    t2 = threshold_otsu(moved)
    b1 = fixed > t1
    b2 = moved > t2
    return {
        'SSIM': np.mean(ssim_vals),
        'NCC': np.mean(ncc_vals),
        'MSE': mean_squared_error(flat_f, flat_m),
        'Pearson': pearsonr(flat_f, flat_m)[0],
        'Hausdorff95': hd95(b1, b2),
        'Dice': dc(b1, b2),
        'MattesMutualInfo': mutual_info_mattes(fixed, moved),
        'OtsuThreshold': (t1 + t2) / 2
    }

# === MAIN LOOP ===
resultados_pre = []  # Metrics before applying deformation
resultados_post = []  # Metrics after applying deformation
tiempos = []  # Inference time per patient

for pid in pacientes_test:
    print(f"Processing patient TRJ_{pid}...")
    start_time = time.time()

    # Define paths for input volumes and deformation fields
    flow_path  = os.path.join(FLOW_DIR, f"flow_TRJ_{pid}.npz")
    dti_path   = os.path.join(HR_DIR, f"dti_{pid}_preprocesado_alta_resolucion.nii")
    perf_path  = os.path.join(HR_DIR, f"perf_{pid}_normalizado_alta_resolucion.nii")

    if not all(os.path.exists(p) for p in [flow_path, dti_path, perf_path]):
        print(f"Missing files for TRJ_{pid}. Skipping.")
        continue

    # Load fixed (perf) and moving (DTI) volumes and deformation field
    dti_img   = nib.load(dti_path)
    perf_img  = nib.load(perf_path)
    dti_data  = dti_img.get_fdata()
    perf_data = perf_img.get_fdata()
    flow      = np.load(flow_path)["flow"]  # Original resolution: (160,192,224,3)

    # --- PRE-REGISTRATION METRICS ---
    m_pre = calc_metrics(perf_data, dti_data)
    m_pre["Patient"] = f"TRJ_{pid}"
    resultados_pre.append(m_pre)

    # --- RESCALING DEFORMATION FIELD ---
    # Interpolates deformation field to match HR volume dimensions
    factors = [t / s for t, s in zip(dti_data.shape, flow.shape[:3])]
    flow_hr = zoom(flow, zoom=factors + [1], order=1)

    # --- APPLY DEFORMATION ---
    # Generate coordinate grid and apply displacement field
    grid = np.meshgrid(
        np.arange(dti_data.shape[0]),
        np.arange(dti_data.shape[1]),
        np.arange(dti_data.shape[2]),
        indexing='ij'
    )
    coords = np.stack(grid, axis=-1).astype(np.float32)
    def_coords = coords + flow_hr
    def_coords = [def_coords[..., i] for i in range(3)]

    # Apply deformation using map_coordinates with 'reflect' mode to minimize boundary artifacts
    # This mode ensures that interpolated values near the image edges are mirrored,
    # which is particularly suitable for medical imaging where anatomical continuity is assumed
    moved = map_coordinates(dti_data, def_coords, order=1, mode='reflect')

    # --- SAVE OUTPUT ---
    vol_out  = os.path.join(VOLS_DIR, f"dti_TRJ_{pid}_registrado_HR.nii")
    flow_out = os.path.join(FLOWS_DIR, f"flow_TRJ_{pid}_HR.npz")
    nib.save(nib.Nifti1Image(moved.astype(np.float32), affine=dti_img.affine), vol_out)
    np.savez_compressed(flow_out, flow=flow_hr)

    # --- POST-REGISTRATION METRICS ---
    m_post = calc_metrics(perf_data, moved)
    m_post["Patient"] = f"TRJ_{pid}"
    resultados_post.append(m_post)

    # Record time
    t_segundos = time.time() - start_time
    tiempos.append(t_segundos)

    # Print detailed metrics
    print(f" TRJ_{pid} — Time: {t_segundos:.2f} seconds")
    print("   PRE-METRICS:")
    for k, v in m_pre.items():
        if k != 'Patient':
            print(f"     {k}: {v:.4f}")
    print("   POST-METRICS:")
    for k, v in m_post.items():
        if k != 'Patient':
            print(f"     {k}: {v:.4f}")
    print()

# === SAVE METRICS ===
df_pre = pd.DataFrame(resultados_pre)
df_post = pd.DataFrame(resultados_post)

df_pre.to_csv(os.path.join(OUT_DIR, "metricas_HR_test_PRE.csv"), index=False)
df_post.to_csv(os.path.join(OUT_DIR, "metricas_HR_test_POST.csv"), index=False)

summary_pre = df_pre.drop(columns=["Patient"]).agg(['mean', 'std']).T
summary_post = df_post.drop(columns=["Patient"]).agg(['mean', 'std']).T
summary_pre.to_csv(os.path.join(OUT_DIR, "resumen_metricas_HR_test_PRE.csv"), header=["mean", "std"])
summary_post.to_csv(os.path.join(OUT_DIR, "resumen_metricas_HR_test_POST.csv"), header=["mean", "std"])

# === SAVE EXECUTION TIME ===
df_tiempos = pd.DataFrame({
    'Patient': [f'TRJ_{pid}' for pid in pacientes_test],
    'Tiempo_segundos': tiempos
})
df_tiempos.to_csv(os.path.join(OUT_DIR, "tiempos_por_paciente.csv"), index=False)

print(f"Mean registration time: {np.mean(tiempos):.2f} seconds")
print(f"Standard deviation:     {np.std(tiempos):.2f} seconds")
