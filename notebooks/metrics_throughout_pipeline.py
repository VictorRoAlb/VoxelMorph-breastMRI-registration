import numpy as np
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from skimage.filters import threshold_otsu
from medpy.metric.binary import dc, hd95
import SimpleITK as sitk

def normalize(img):
    return (img - img.min()) / (img.max() - img.min())

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

def calc_metrics(vol1, vol2):
    v1, v2 = normalize(vol1), normalize(vol2)

    ssim_vals = [
        ssim(v1[:, :, z], v2[:, :, z], data_range=1)
        for z in range(v1.shape[2])
        if v1[:, :, z].std() > 0 and v2[:, :, z].std() > 0
    ]

    ncc_vals = [
        np.corrcoef(v1[:, :, z].ravel(), v2[:, :, z].ravel())[0, 1]
        for z in range(v1.shape[2])
        if v1[:, :, z].std() > 0 and v2[:, :, z].std() > 0
    ]

    mse_val = mean_squared_error(v1.ravel(), v2.ravel())
    mi_val = mutual_info_mattes(v1, v2)

    t1 = threshold_otsu(v1)
    t2 = threshold_otsu(v2)
    b1, b2 = v1 > t1, v2 > t2

    return {
        'SSIM': np.mean(ssim_vals),
        'NCC': np.mean(ncc_vals),
        'MSE': mse_val,
        'Dice': dc(b1, b2),
        'Hausdorff95': hd95(b1, b2),
        'MattesMutualInfo': mi_val,
        'OtsuThreshold': (t1 + t2) / 2
    }
