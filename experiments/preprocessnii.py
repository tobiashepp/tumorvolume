from io import BytesIO
from zipfile import ZipFile
import gzip
from pathlib import Path
import argparse

import numpy as np
import nilearn.image
import nibabel as nib
from midastools.pet.isocont import isocont
from midastools.misc.orientation import reorient_nii

from joblib import Parallel, delayed
import h5py


def normalization(arr, arr_min=-1000, arr_max=1000):
    """
    Normalizes the array data to [0, 1] interval.
    Args:
        arr (array.pyi): The given array.
        arr_min (int): The minimum value.
        arr_max (int): The maximum value.

    Returns: array with normalized values.

    """
    arr -= arr_min
    arr *= (1 / (arr_max - arr_min))
    arr = np.clip(arr, 0, 1)
    return arr


def run_preprocessing(ct_image,
                      petsuv_image,
                      orientation=("L", "A", "S"),
                      target_spacing=(2, 2, 3),
                      xy_shape=200,
                      ct_min_max=(-1000, 1000),
                      pet_min_max=(0, 40),
                      isocont_perc_threshold=25):
    """
    Reads nii.gz-files in a ziparchive archive and resamples the images, storing them in a hdf5-file.

    Args:
        ct_image (str): String representing a path to a zip archive containing nii.gz.
        petsuv_image (str): String representing a path to a directory where the output will be stored.
        orientation (str, str, str): Tuple of strings specifying the desired orientation following radiological
                                     Terminology. Default: ("L", "A", "S").
        target_spacing (int, int, int): Tuple of integers specifying the desired spacing used in the affine matrix
                                        converting voxel position to physical position (mm). Default: (2, 2, 3)
        xy_shape (int): Integer specifying the array shape in x and y direction.
        ct_min_max (tuple of int): Minimum and maximum value for normalization.
        pet_min_max (tuple of int): Minimum and maximum value for normalization.
        isocont_perc_threshold (float): Float representing the percentile threshold used in isocont function.
                                        Default: 25.
    Returns:
        numpy array [2, xy_shape, xy_shape, xy_shape] where [0, :, :, :] is the preprocessed ct and [1, :, :, :] is the
        preprocessed petSUV.
    """

    ct_img = nib.load(ct_image)
    or_ct_img = reorient_nii(ct_img, target_orientation=orientation)
    pet_img = nib.load(petsuv_image)
    or_pet_img = reorient_nii(ct_img, target_orientation=orientation)

    # resample to target resolution
    orig_spacing = np.array(or_ct_img.header.get_zooms())
    orig_shape = or_ct_img.header.get_data_shape()
    target_affine = np.copy(or_ct_img.affine)
    target_affine[:3, :3] = np.diag(target_spacing / orig_spacing) @ or_ct_img.affine[:3, :3]
    target_shape = (xy_shape, xy_shape, int(orig_shape[2] * (orig_spacing[2] / target_spacing[2])))

    print("original: ", orig_spacing, orig_shape)
    print("target: ", target_spacing, target_shape)

    rs_ct_img = nilearn.image.resample_img(or_ct_img,
                                           target_affine=target_affine,
                                           target_shape=target_shape,
                                           interpolation="continuous",
                                           fill_value=ct_min_max[0])
    rs_pet_img = nilearn.image.resample_to_img(or_pet_img,
                                               rs_ct_img,
                                               interpolation="continuous",
                                               fill_value=pet_min_max[0])

    # cast and concat
    ct = normalization(rs_or_ct_img.get_fdata(), ct_min_max[0], ct_min_max[1])
    petsuv = normalization(rs_pet_img.get_fdata(), pet_min_max[0], pet_min_max[1])
    ret_img = np.stack([petsuv, ct], axis=0).astype(np.float16)
    print(ret_img.shape)
    ret_mask = np.array(rs_img_dict["mask"].get_fdata() > 0).astype(np.uint8)
    ret_mask = ret_mask[np.newaxis, :]
    print(ret_mask.shape)
    ret_mask_iso = np.array(rs_img_dict["mask_iso"].get_fdata() > 0).astype(np.uint8)
    ret_mask_iso = ret_mask_iso[np.newaxis, :]
    print(ret_mask_iso.shape)
    ret_affine = rs_or_ct_img.affine
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", help="path to ct as nii or nii.gz", required=True)
    parser.add_argument("--pet", help="path to petSUV as nii or nii.gz", required=True)
    parser.add_argument("--xy", help="desired shape", required=True)
    parser.add_argument("--ort", help="desired orientation", default=("L", "A", "S"))
    parser.add_argument("--spc", help="desired spacing", default=(2, 2, 3))
    parser.add_argument("--isc", help="percentile threshold for isocont", default=25)

    args = parser.parse_args()

    run_preprocessing(ct_image=args.ct,
                      petsuv_image=args.pet,
                      xy_shape=args.xy,
                      orientation=args.ort,
                      target_spacing=args.spc,
                      isocont_perc_threshold=args.isc)


if __name__ == '__main__':
    main()
