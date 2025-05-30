# DESCRIPTION
# Script to resample high-resolution DTI and perfusion volumes to a uniform shape ( [432 432 720]).
# Originally used to resample volumes to low resolution (160x192x224) for model input,
# and later reused to match high-resolution voxel dimensions for deformation field rescaling.
# The core code is identical in both cases - only the target shape and output directories change.

# CONFIGURATION 
# Base input and output directories - adapt according to your project structure.
inputBase   = 'D:\Imagenes DICOM TRJ';
outputBase  = 'C:\TFG Victor\TFG\resolucion_original';
mkdir(outputBase);

# Target resolution - adjust depending on whether you need low or high resampling.
targetSize  = [432 432 720];

# MAIN LOOP 
# Loop through all patient IDs. In this specific case patients are indexed from 1 to 60,
# excluding 9 and 22 due to missing or corrupted data. Adjust this list to match your dataset.
tic;
for i = 1:60
    if ismember(i, [9 22]), continue; end
    pacienteID = sprintf('TRJ_%02d', i);
    fprintf('\nProcessing %s...\n', pacienteID);

    #  RESAMPLE PERFUSION VOLUME 
    try
        perfDir = fullfile(inputBase, pacienteID, 'perf');

        # Disable warnings temporarily while reading DICOM series
        prev = warning('off','all');
        [volPerf, ~] = dicomreadVolume(perfDir);
        warning(prev);  # Restore warning settings

        volPerf = squeeze(volPerf);  # Remove singleton dimensions

        # Resize using linear interpolation - this can be adjusted to other methods
        volPerfResized = imresize3(volPerf, targetSize, 'linear');

        # Save as NIfTI
        filenamePerf = fullfile(outputBase, sprintf('perf_%02d_TRJ_reso_original.nii', i));
        niftiwrite(volPerfResized, filenamePerf);
        fprintf('Perfusion saved: %s\n', filenamePerf);

        # Reload to confirm shape
        volSaved = niftiread(filenamePerf);
        fprintf('Saved perfusion dimensions: [%d %d %d]\n', size(volSaved));
    catch ME
        warning(prev);
        fprintf('Error in perfusion of %s: %s\n', pacienteID, ME.message);
    end

    #  RESAMPLE DTI VOLUME 
    try
        dtiDir = fullfile(inputBase, pacienteID, 'dti');
        prev = warning('off','all');
        [volDTI, ~] = dicomreadVolume(dtiDir);
        warning(prev);

        volDTI = squeeze(volDTI);

        # Resize using linear interpolation - could be changed if needed
        volDTIResized = imresize3(volDTI, targetSize, 'linear');

        # Save as NIfTI
        filenameDTI = fullfile(outputBase, sprintf('dti_%02d_TRJ_reso_original.nii', i));
        niftiwrite(volDTIResized, filenameDTI);
        fprintf('DTI saved: %s\n', filenameDTI);

        # Reload to confirm shape
        volSaved = niftiread(filenameDTI);
        fprintf('Saved DTI dimensions:       [%d %d %d]\n', size(volSaved));
    catch ME
        warning(prev);
        fprintf('Error in DTI of %s: %s\n', pacienteID, ME.message);
    end
end

# Report total runtime
totalTime = toc;
fprintf('\nTotal time: %.2f s\n', totalTime);
