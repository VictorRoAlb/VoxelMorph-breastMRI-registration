
# VoxelMorph Applied to Breast MRI Registration
This repository contains the full implementation and documentation of my Bachelor's Thesis in Biomedical Engineering at ETSII - Universitat Politècnica de València (UPV). The project focuses on adapting the VoxelMorph deep learning-based deformable registration framework to dynamic contrast-enhanced (DCE) and diffusion tensor imaging (DTI) breast MRI data obtained in a real clinical setting.

## Project Overview
The aim of this project is to prepare and adapt breast MRI data for deformable multimodal registration using VoxelMorph. A complete preprocessing pipeline was developed, including dimensionality reduction, intensity normalization, and rigid/affine registrations. Pretrained 3D VoxelMorph models were fine-tuned using both MSE and NCC loss functions on downsampled image pairs. Final deformation fields were rescaled to high-resolution and applied to the original volumes. Quantitative and qualitative assessments were carried out using multiple similarity metrics to evaluate registration performance before and after processing. Qualitative evaluation was performed by visualizing the 3D volumes using MATLAB-based tools.

A **workflow diagram** illustrating the main processing steps from data acquisition to evaluation is included in this repository for clarity and ease of understanding.

## Repository Structure

- `Matlab/` – Auxiliary MATLAB scripts for volume resampling and multimodal visualization
- `notebooks/notebooks/` – Python scripts for training, evaluation, and metric computation
- `workflow_diagram.png` – Diagram illustrating the main processing steps of the project
- `mse_perf_dti_final_weights.h5` – Final VoxelMorph model weights trained with MSE loss
- `perf_dti_final_weights_NCC.h5` – Final VoxelMorph model weights trained with NCC loss
- `README.md` – Main project documentation


## Technologies Used
- Python 3.x  
- TensorFlow 2.18.0  
- VoxelMorph 0.2  
- SimpleITK  
- nibabel, numpy, pandas, scikit-image, scipy, medpy
- matplotlib
- tqdm
- multiprocessing
- MATLAB:
  - `imresize3` for 3D downsampling and inter-modality resolution matching
  - `vol3d` and `volumeViewer` for 3D visualization
  - `imshowpair` and RGB overlay methods for 2D multimodal comparison

## Credits
This project includes adaptations of code from VoxelMorph (https://github.com/voxelmorph/voxelmorph), developed by the Computational Radiology Lab (Harvard/MIT), released under the MIT License.

## Contact
Victor Rodríguez Albendea 
Bachelor's Degree in Biomedical Engineering  
Email: victorrodriguezalb@gmail.com


