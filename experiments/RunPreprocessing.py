import re
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


def save_nii(project_name, destination_dir, patient, data_dict):
    """

    Args:
        project_name (str): String representing the project name.
        destination_dir (str): String representing a path to a directory where the output will be stored.
        patient (str): Patient name.
        data_dict (dict): Dictionary including arrays for the patients ct, petsuv, mask and affine.

    Returns: None

    """
    out_dir = Path(destination_dir)
    new_project_dir = out_dir / project_name
    new_project_dir.mkdir(exist_ok=True)

    for key, file in data_dict.items():

        new_pat_dir = new_project_dir / patient
        new_pat_dir.mkdir(exist_ok=True)
        filename = patient + "_" + key + ".nii.gz"
        outfile = new_pat_dir / filename
        nib.save(file, outfile)


def run_preprocessing(zip_archive,
                      destination_dir,
                      orientation=("L", "A", "S"),
                      target_spacing=(1, 1, 3),
                      xy_shape=200,
                      ct_min_max=(-1000, 1000),
                      pet_min_max=(0, 40),
                      isocont_perc_threshold=25,
                      jobs=10,
                      save_as_nii=False):
    """
    Reads nii.gz-files in a ziparchive archive and resamples the images, storing them in a hdf5-file.

    Args:
        zip_archive (str): String representing a path to a zip archive containing nii.gz.
        destination_dir (str): String representing a path to a directory where the output will be stored.
        orientation (str, str, str): Tuple of strings specifying the desired orientation following radiological
                                     Terminology. Default: ("L", "A", "S").
        target_spacing (int, int, int): Tuple of integers specifying the desired spacing used in the affine matrix
                                        converting voxel position to physical position (mm). Default: (2, 2, 3)
        xy_shape (int): Integer specifying the array shape in x and y direction.
        ct_min_max (tuple of int): Minimum and maximum value for normalization.
        pet_min_max (tuple of int): Minimum and maximum value for normalization.
        isocont_perc_threshold (float): Float representing the percentile threshold used in isocont function.
                                        Default: 25.
        jobs (int): Number of parallel jobs. Defaults: 5.
        save_as_nii (bool): Determines, whether the data is also stored as nii_gz, e.g. for viewing purposes.
    Returns:
        None
    """

    zip_archive = Path(zip_archive)
    project_name = zip_archive.stem

    with ZipFile(zip_archive, "r") as zf:
        # Get a list of all directories and files stored in zip archive
        complete_name_list = zf.namelist()

    # find a pet for every patient
    petsuv_re = re.compile(".*petsuv.nii.gz")
    petsuv_list = list(filter(petsuv_re.match, complete_name_list))
    # append the ct- and mask-files belonging to the patient
    patients = {}
    for petsuv in petsuv_list:
        files = {"ct": None, "petsuv": None, "mask": None}
        petsuv_path = Path(petsuv)
        patient = str(petsuv_path.parent).split("/")[-1]
        files["ct"] = petsuv.replace("petsuv", "ct")
        files["petsuv"] = petsuv
        files["mask"] = petsuv.replace("petsuv", "mask")
        patients[patient] = files

    def proc(patient):
        print(patient)
        img_dict = {}
        with ZipFile(zip_archive, "r") as zip_obj:
            for series in patients[patient]:
                print(series)
                file = patients[patient][series]
                # open the zipfile and read gz-files as bytes. Gzip can reconstruct the gz-files and read them as bytes.
                # The content is used to reconstruct the images.
                with zip_obj.open(file, "r") as zfile:
                    gzfile = gzip.GzipFile(fileobj=BytesIO(zfile.read()), mode="rb")
                    fh = nib.FileHolder(fileobj=BytesIO(gzfile.read()))
                    img_file = nib.Nifti1Image.from_file_map({"header": fh, "image": fh, "affine": fh})
                    img_dict[series] = reorient_nii(img_file, target_orientation=orientation)

        # resample to target resolution
        rs_img_dict = {}
        orig_spacing = np.array(img_dict["ct"].header.get_zooms())
        orig_shape = img_dict["ct"].header.get_data_shape()
        target_affine = np.copy(img_dict["ct"].affine)
        target_affine[:3, :3] = np.diag(target_spacing / orig_spacing) @ img_dict["ct"].affine[:3, :3]
        target_shape = (xy_shape, xy_shape, int(orig_shape[2] * (orig_spacing[2] / target_spacing[2])))

        print("original: ", orig_spacing, orig_shape)
        print("target: ", target_spacing, target_shape)

        rs_img_dict["ct"] = nilearn.image.resample_img(img_dict["ct"],
                                                       target_affine=target_affine,
                                                       target_shape=target_shape,
                                                       interpolation="continuous",
                                                       fill_value=-1024)
        rs_img_dict["petsuv"] = nilearn.image.resample_to_img(img_dict["petsuv"],
                                                              rs_img_dict["ct"],
                                                              interpolation="continuous",
                                                              fill_value=0)
        rs_img_dict["mask"] = nilearn.image.resample_to_img(img_dict["mask"],
                                                            rs_img_dict["ct"],
                                                            interpolation="nearest",
                                                            fill_value=0)
        rs_img_dict["mask_iso"] = nib.Nifti1Image(isocont(img_arr=rs_img_dict["petsuv"].get_fdata(),
                                                          mask_arr=(rs_img_dict["mask"].get_fdata() > 0).astype(np.uint8),
                                                          percentile_threshold=isocont_perc_threshold),
                                                          rs_img_dict["mask"].affine)

        # cast and concat
        ct = normalization(rs_img_dict["ct"].get_fdata(), ct_min_max[0], ct_min_max[1])
        petsuv = normalization(rs_img_dict["petsuv"].get_fdata(), pet_min_max[0], pet_min_max[1])
        ret_img = np.stack([petsuv, ct], axis=0).astype(np.float16)
        print(ret_img.shape)
        ret_mask = np.array(rs_img_dict["mask"].get_fdata() > 0).astype(np.uint8)
        ret_mask = ret_mask[np.newaxis, :]
        print(ret_mask.shape)
        ret_mask_iso = np.array(rs_img_dict["mask_iso"].get_fdata() > 0).astype(np.uint8)
        ret_mask_iso = ret_mask_iso[np.newaxis, :]
        print(ret_mask_iso.shape)
        ret_affine = rs_img_dict["ct"].affine

        if save_as_nii:
            save_nii(project_name=project_name,
                     destination_dir=destination_dir,
                     patient=patient,
                     data_dict=rs_img_dict)
        return_dict = {"patient": patient,
                       "image": ret_img,
                       "mask": ret_mask,
                       "mask_iso": ret_mask_iso,
                       "affine": ret_affine}
        return return_dict

    def chunky(lst, jobs):
        """Yield successive n-sized chunks from lst."""
        for di in range(0, len(lst), jobs):
            if not (i + jobs) > len(lst):
                so_chunky = lst[i:i + jobs]
            else:
                so_chunky = lst[i:len(lst)]
            yield so_chunky

    patient_chunks = list(chunky(list(patients.keys()), jobs))
    for chunk in patient_chunks:
        results = Parallel(n_jobs=min(jobs, len(chunk)))(delayed(proc)(patient=patient) for patient in chunk)

        out_dir = Path(destination_dir)
        hdf5name = project_name + "_3D.h5"
        outfile = out_dir/hdf5name
        with h5py.File(outfile, "a") as hdf5:
            for result in results:
                for item in ["image", "mask", "mask_iso"]:
                    affine_to_list = result["affine"].tolist()
                    group = hdf5.require_group(item)
                    group.require_dataset(name=result["patient"],
                                          data=result[item],
                                          shape=result[item].shape,
                                          dtype=result[item].dtype)
                    group[result["patient"]].attrs["affine"] = affine_to_list
                    group[result["patient"]].attrs["project"] = project_name

    return "done"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc", help="zip-archive with nii.gz-images", required=True)
    parser.add_argument("--dst", help="output directory", required=True)
    parser.add_argument("--ort", help="desired orientation", default=("L", "A", "S"))
    parser.add_argument("--spc", help="desired spacing", default=(2, 2, 3))
    parser.add_argument("--isc", help="percentile threshold for isocont", default=25)
    parser.add_argument("--jbs", help="parallel jobs", default=10)
    parser.add_argument("--sav", help="create nii.gz", default=False)

    args = parser.parse_args()

    run_preprocessing(zip_archive=args.arc,
                      destination_dir=args.dst,
                      orientation=args.ort,
                      target_spacing=args.spc,
                      isocont_perc_threshold=args.isc,
                      save_as_nii=args.sav)


if __name__ == '__main__':
    main()
