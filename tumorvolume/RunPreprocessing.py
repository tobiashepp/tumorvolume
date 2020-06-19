import re
from io import BytesIO
from zipfile import ZipFile
import gzip
from pathlib import Path
import argparse

import numpy as np
import scipy
import nilearn.image
import nibabel as nib
from midastools.pet.isocont import isocont
from midastools.misc.orientation import reorient_nii


def preprocess_image(img,
                     target_spacing,
                     target_shape=None,
                     target_orientation=('L', 'A', 'S'),
                     interpolation="continuous",
                     fill_value=0,
                     dtype=np.float16):
    """Rescale and reorient nifti image.

    Args:
        img (nib.NiftiImage): Original nifti image
        target_spacing (tuple): Target image spacing
        target_shape (tuple, optional): Target image shape. Defaults to None.
        target_orientation (tuple, optional): Target image orientation. Defaults to ('L','A','S').
        interpolation (str, optional): Interpolation type (nilearn.image.resample). Defaults to "continuous".
        fill_value (int, optional): Fill value. Defaults to 0.
        dtype (type, optional): Data type for the resampled image. Defaults to np.float16.

    Returns:
        nib.NiftiImage: Processed nifti image
    """

    # reorient to target orientation
    img = reorient_nii(img, target_orientation)
    # resample to target resolution
    orig_spacing = np.array(img.header.get_zooms())
    target_affine = np.copy(img.affine)
    target_affine[:3, :3] = np.diag(target_spacing / orig_spacing) @ img.affine[:3, :3]
    resampled_img = nilearn.image.resample_img(img,
                                               target_affine=target_affine,
                                               target_shape=target_shape,
                                               interpolation=interpolation,
                                               fill_value=fill_value)
    resampled_img.set_data_dtype(dtype)

    # normalization
    img_data = resampled_img.get_fdata().astype(dtype)
    p = np.percentile(img_data, [5, 95])
    img_data = (img_data - p[0]) / (p[1] - p[0])
    img_data = np.clip(img_data, 0.0, 1.0)
    return img_data.astype(dtype), resampled_img.affine


def preprocess_label(label,
                     target_spacing,
                     target_shape=None,
                     target_orientation=('L', 'A', 'S'),
                     fill_value=0,
                     dtype=np.uint8):
    """Rescale and reorient nifti image.

    Args:
        label (nib.NiftiImage): Original nifti image
        target_spacing (tuple): Target image spacing
        target_shape (tuple, optional): Target image shape. Defaults to None.
        target_orientation (tuple, optional): Target image orientation. Defaults to ('L','A','S').
        fill_value (int, optional): Fill value. Defaults to 0.
        dtype (type, optional): Data type for the resampled image. Defaults to np.uint8.

    Returns:
        tuple(label data as an array, label affine as an array)
    """

    # reorient to target orientation
    label = reorient_nii(label, target_orientation)
    # resample to target resolution
    orig_spacing = np.array(label.header.get_zooms())
    target_affine = np.copy(label.affine)
    target_affine[:3, :3] = np.diag(target_spacing / orig_spacing) @ label.affine[:3, :3]
    resampled_label = nilearn.image.resample_img(label,
                                                 target_affine=target_affine,
                                                 target_shape=target_shape,
                                                 interpolation="nearest",
                                                 fill_value=fill_value)
    resampled_label.set_data_dtype(dtype)

    # fill holes and keep only largest connected component
    resampled_label = nilearn.image.largest_connected_component_img(resampled_label)
    label_data = resampled_label.get_fdata().astype(dtype)
    label_data = scipy.ndimage.morphology.binary_fill_holes(label_data)
    return label_data.astype(dtype), resampled_label.affine


def run_preprocessing(archive, destination_dir, orientation, spacing, isocont_perc_threshold):
    """
    Reads nii.gz-files in a ziparchive archive and resamples the images, storing them as nii.gz in the destination_dir.

    Args:
        archive (str): String representing a path to a zip archive containing nii.gz
        destination_dir (str): String representing a path to a directory where the output will be stored.

    Returns:
        None
    """

    out_dir = Path(destination_dir)
    file = Path(archive)

    project_name = file.stem
    new_project_dir = out_dir/project_name
    new_project_dir.mkdir(exist_ok=True)
    with ZipFile(file, "r") as zf:
        # Get a list of all directories and files stored in zip archive
        complete_name_list = zf.namelist()

        files = {"petsuv": [], "ct": [], "mask": []}
        # find a pet for every patient
        petsuv_re = re.compile(".*petsuv.nii.gz")
        petsuv_list = list(filter(petsuv_re.match, complete_name_list))
        files["petsuv"] = petsuv_list
        # append the ct- and mask-files belonging to the patient
        for pet in files["petsuv"]:
            files["ct"].append(pet.replace("petsuv", "ct"))
            files["mask"].append(pet.replace("petsuv", "mask"))

        for key in files:
            for file in files[key][0]:
                file_path = Path(file)
                patient = str(file_path.parent).split("/")[-1]
                new_pat_dir = new_project_dir / patient
                print(patient, key)
                if key == "petsuv":
                    new_pat_dir.mkdir(exist_ok=True)
                # open the zipfile and read gz-files as bytes. Gzip can reconstruct the gz-files and read them as bytes.
                # The content is used to reconstruct the images.
                with zf.open(file, "r") as zfile:
                    gzfile = gzip.GzipFile(fileobj=BytesIO(zfile.read()), mode="rb")
                    fh = nib.FileHolder(fileobj=BytesIO(gzfile.read()))
                    img_file = nib.Nifti1Image.from_file_map({"header": fh, "image": fh, "affine": fh})
                    source_shape = img_file.header.get_data_shape()
                    print(source_shape)
                    source_spacing = img_file.header.get_zooms()
                    print(source_spacing, source_shape)
                    target_spacing = spacing
                    target_shape = (512, 512, int(source_shape[2] * (source_spacing[2] / target_spacing[2])))
                    if key == "petsuv" or key == "ct":
                        resampled_img = preprocess_image(img=img_file,
                                                         target_spacing=target_spacing,
                                                         target_shape=target_shape,
                                                         target_orientation=orientation,
                                                         interpolation="continuous",
                                                         fill_value=0,
                                                         dtype=np.float32)
                        if key == "petsuv":
                            petsuv_data = resampled_img[0]
                    if key == "mask":
                        resampled_lbl_data = preprocess_label(label=img_file,
                                                              target_spacing=target_spacing,
                                                              target_shape=target_shape,
                                                              target_orientation=orientation,
                                                              fill_value=0,
                                                              dtype=np.float32)
                        resampled_lbl_data_isocont = isocont(img_arr=petsuv_data,
                                                             mask_arr=resampled_lbl_data_isocont[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc", help="zip-archive with nii.gz-images", required=True)
    parser.add_argument("--dst", help="output directory", required=True)
    parser.add_argument("--ort", help="desired orientation", default=("L", "A", "S"))
    parser.add_argument("--spc", help="desired spacing", default=(2, 2, 3))
    parser.add_argument("--isc", help="percentile threshold for isocont", default=25)

    args = parser.parse_args()
    archive = args.arc
    destination_dir = args.dst
    orientation = args.ort
    target_spacing = args.spc
    isocontour_perc_threshold = args.isc

    run_preprocessing(archive=archive,
                      destination_dir=destination_dir,
                      orientation=orientation,
                      spacing=target_spacing,
                      isocont_perc_threshold=isocontour_perc_threshold)


if __name__ == '__main__':
    main()
