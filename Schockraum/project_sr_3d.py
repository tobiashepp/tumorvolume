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


def normalization(arr, arr_min=0, arr_max=100):
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


def run_preprocessing(project_dir,
                      destination_dir,
                      orientation=("L", "A", "S"),
                      target_spacing=(1, 1, 1),
                      xy_shape=200,
                      ct_min_max=(0, 100),
                      jobs=10):
    """
    Reads nii-files in and resamples the images, storing them in a hdf5-file.

    Args:
        project_dir (str): String representing a path to a zip archive containing nii.gz.
        destination_dir (str): String representing a path to a directory where the output will be stored.
        orientation (str, str, str): Tuple of strings specifying the desired orientation following radiological
                                     Terminology. Default: ("L", "A", "S").
        target_spacing (int, int, int): Tuple of integers specifying the desired spacing used in the affine matrix
                                        converting voxel position to physical position (mm). Default: (2, 2, 3)
        xy_shape (int): Integer specifying the array shape in x and y direction.
        ct_min_max (tuple of int): Minimum and maximum value for normalization.
        jobs (int): Number of parallel jobs. Defaults: 10.
    Returns:
        None
    """
    project_name = Path(project_dir).name
    dirs_to_draw = []
    for study_dir in project_dir.iterdir():
        dirs_to_draw.append(str(study_dir))

    print("total: " + str(len(list(project_dir.iterdir())) - 3) + "\n" + "first: " + dirs_to_draw[0] + "\n" + "last: " +
          dirs_to_draw[-1])

    def proc(patient_dir):
        try:
            patient = Path(patient_dir).name
            print(patient)
            ct_img = nib.load(patient_dir/(patient + "_ants_reg_strip.nii"))
            or_ct_img = reorient_nii(ct_img, target_orientation=orientation)

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
                                                   fill_value=0)
            # cast and concat
            ct = normalization(rs_ct_img.get_fdata(), ct_min_max[0], ct_min_max[1])
            return patient, ct
        except Exception as e:
            patient = Path(patient_dir).name
            print(e)
            error_dict["pat_hash"].append(patient)
            error_dict["cause"].append(e)

    def chunky(lst, jbs):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), jbs):
            if not (i + jbs) > len(lst):
                so_chunky = lst[i:i + jbs]
            else:
                so_chunky = lst[i:len(lst)]
            yield so_chunky

    error_dict = {"layer": [], "pat_hash": [], "cause": []}
    patient_chunks = list(chunky(dirs_to_draw, jobs))
    for chunk in patient_chunks:
        results = Parallel(n_jobs=min(jobs, len(chunk)))(delayed(proc)(patient_dir=patient_dir)
                                                         for patient_dir in chunk)

        out_dir = Path(destination_dir)
        hdf5name = project_name + "_3D.h5"
        outfile = out_dir/hdf5name
        with h5py.File(outfile, "a") as hdf5:
            for result in results:
                affine_to_list = result[1].tolist()
                group = hdf5.require_group("image")
                group.require_dataset(name=result[0],
                                      data=result[1],
                                      shape=result[1].shape,
                                      dtype=result[1].dtype)
                group[result[0]].attrs["affine"] = affine_to_list
                group[result[0]].attrs["project"] = project_name

    return "done"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prj", help="zip-archive with nii.gz-images", required=True)
    parser.add_argument("--dst", help="output directory", required=True)
    parser.add_argument("--ort", help="desired orientation", default=("L", "A", "S"))
    parser.add_argument("--spc", help="desired spacing", default=(1, 1, 1))
    parser.add_argument("--jbs", help="parallel jobs", default=20)

    args = parser.parse_args()

    run_preprocessing(project_dir=args.prj,
                      destination_dir=args.dst,
                      orientation=args.ort,
                      target_spacing=args.spc)


if __name__ == '__main__':
    main()
