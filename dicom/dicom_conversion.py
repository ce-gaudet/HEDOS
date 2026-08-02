"""
This file is adapted from the rt-utils project:
https://github.com/ethanio12345/hedos

Modifications:
- Contours were changed to filled polygons
- Bypass of degenerate polygons
"""

import os
import numpy as np
import pydicom
import SimpleITK as sitk
from dicom.rt_utils.rt_utils_mod import RTStructBuilder

# NOTE:
# RTSTRUCT handling in this pipeline relies on a locally modified rt-utils
# implementation to ensure robust contour-to-mask conversion
# (collapsed polygon handling).


def _norm(name: str) -> str:
    """Normalize ROI names so downstream lookups are consistent."""
    return name.strip().lower().replace(" ", "_")

def load_dicom_series(directory: str) -> sitk.Image:
    """Load a DICOM CT series using SimpleITK."""
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(directory)
    if not series_IDs:
        raise ValueError(f"No DICOM series found in directory: {directory}")

    series_file_names = reader.GetGDCMSeriesFileNames(directory, series_IDs[0])
    reader.SetFileNames(series_file_names)
    return reader.Execute()


def load_dose_image(rtdose_path: str) -> sitk.Image:
    """Load RTDOSE as a SimpleITK image with spacing and origin."""
    dose_ds = pydicom.dcmread(rtdose_path)

    dose_array = dose_ds.pixel_array.astype(np.float32)
    scaling = float(getattr(dose_ds, "DoseGridScaling", 1.0))
    dose_array *= scaling

    dose_image = sitk.GetImageFromArray(dose_array)

    if hasattr(dose_ds, "GridFrameOffsetVector") and len(dose_ds.GridFrameOffsetVector) > 1:
        z_spacing = float(
            dose_ds.GridFrameOffsetVector[1] - dose_ds.GridFrameOffsetVector[0]
        )
    else:
        z_spacing = 1.0

    px, py = [float(x) for x in dose_ds.PixelSpacing]
    dose_image.SetSpacing((py, px, z_spacing))

    if hasattr(dose_ds, "ImagePositionPatient"):
        dose_image.SetOrigin([float(v) for v in dose_ds.ImagePositionPatient])

    return dose_image


def resample_to_reference(
    image: sitk.Image, reference: sitk.Image, is_label: bool
) -> sitk.Image:
    """Resample an image onto a reference grid."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    )
    resampler.SetOutputPixelType(image.GetPixelID())
    return resampler.Execute(image)


def extract_structures(
    rtstruct_path: str, ct_series_dir: str, ct_image: sitk.Image
) -> dict:
    """Rasterize RTSTRUCT ROIs as masks aligned to the CT grid."""
    rtb = RTStructBuilder.create_from( #Using Rtstruct builder, with removing collapsed polygons
        dicom_series_path=ct_series_dir,
        rt_struct_path=rtstruct_path,
    )

    structure_masks = {}

    for roi_name in rtb.get_roi_names():
        mask_np = rtb.get_roi_mask_by_name(roi_name)
        if mask_np is None or mask_np.sum() == 0:
            continue

        name = _norm(roi_name)
        mask_img = sitk.GetImageFromArray(mask_np.astype(np.uint8))
        mask_img.SetSpacing(ct_image.GetSpacing())
        mask_img.SetOrigin(ct_image.GetOrigin())
        mask_img.SetDirection(ct_image.GetDirection())

        structure_masks[name] = mask_img

    return structure_masks


def save_hedos_inputs(
    ct_image: sitk.Image,
    structure_masks: dict,
    dose_image: sitk.Image,
    output_dir: str,
) -> None:
    """Save HEDOS-ready NumPy inputs."""
    os.makedirs(output_dir, exist_ok=True)

    affine = np.eye(4, dtype=np.float64)
    spacing = np.array(ct_image.GetSpacing())
    direction = np.array(ct_image.GetDirection()).reshape(3, 3)
    origin = np.array(ct_image.GetOrigin())

    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin

    dose_array = sitk.GetArrayFromImage(dose_image).astype(np.float32)
    dose_array = np.transpose(dose_array, (1, 2, 0))

    np.save(os.path.join(output_dir, "dose.npy"), dose_array)
    np.save(os.path.join(output_dir, "affine.npy"), affine)

    seg_arrays = {
        organ: sitk.GetArrayFromImage(mask).astype(np.uint8)
        for organ, mask in structure_masks.items()
    }

    np.savez_compressed(
        os.path.join(output_dir, "compressed_segs.npz"), **seg_arrays
    )

    print("[HEDOS] Files written to:", os.path.abspath(output_dir))
    print("[HEDOS] Number of ROIs:", len(seg_arrays))


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HEDOS_ROOT = os.path.dirname(_THIS_DIR)
OUTPUT_DIR = os.path.join(_HEDOS_ROOT, "input", "patient")


def dicom_conversion(
    CT_DIR: str,
    RTSTRUCT_PATH: str,
    RTDOSE_PATH: str,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Convert CT + RTSTRUCT + RTDOSE to HEDOS NumPy inputs."""
    ct_image = load_dicom_series(CT_DIR)
    dose_image = load_dose_image(RTDOSE_PATH)

    structure_masks = extract_structures(RTSTRUCT_PATH, CT_DIR, ct_image)
    dose_on_ct = resample_to_reference(dose_image, ct_image, is_label=False)

    save_hedos_inputs(ct_image, structure_masks, dose_on_ct, output_dir)

    print("[HEDOS] NumPy conversion done")
