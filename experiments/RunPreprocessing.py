from midasvessel.processing.preprocessing import preprocess_image, preprocess_label
import re
import nibabel as nib
import subprocess
import numpy as np
from io import BytesIO
from zipfile import ZipFile
import gzip
from pathlib import Path
import argparse


def run_preprocessing(archive, destination_dir):
    """
    Reads nii.gz-files in a ziparchive archive and resamples the images, storing them as nii.gz in the destination_dir.
    Args:
        archive:
        destination_dir:

    Returns:

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
        petsuv_re = re.compile(".*petsuv.nii.gz")
        petsuv_list = list(filter(petsuv_re.match, complete_name_list))
        files["petsuv"] = petsuv_list
        for pet in files["petsuv"]:
            files["ct"].append(pet.replace("petsuv", "ct"))
            files["mask"].append(pet.replace("petsuv", "mask"))

        for key in files:
            for file in files[key][:2]:
                file_path = Path(file)
                patient = str(file_path.parent).split("/")[-1]
                new_pat_dir = new_project_dir / patient
                print(patient)
                if key == "petsuv":
                    new_pat_dir.mkdir(exist_ok=True)
                with zf.open(file, "r") as zfile:
                    gzfile = gzip.GzipFile(fileobj=BytesIO(zfile.read()), mode="rb")
                    fh = nib.FileHolder(fileobj=BytesIO(gzfile.read()))
                    img_file = nib.Nifti1Image.from_file_map({"header": fh, "image": fh, "affine": fh})
                    source_shape = img_file.header.get_data_shape()
                    print(source_shape)
                    source_spacing = img_file.header.get_zooms()
                    print(source_spacing, source_shape)
                    target_spacing = (2, 2, 3)
                    target_shape = (512, 512, int(source_shape[2] * (source_spacing[2] / target_spacing[2])))
                    if key == "petsuv" or key == "ct":
                        resampled_img = preprocess_image(img=img_file,
                                                         target_spacing=target_spacing,
                                                         target_shape=target_shape,
                                                         target_orientation=('L', 'A', 'S'),
                                                         interpolation="continuous",
                                                         fill_value=0,
                                                         dtype=np.float32)
                        resampled_filename = patient + "_" + key + "_resampled" + ".nii"
                        outfile = new_pat_dir/resampled_filename
                        nib.save(resampled_img, outfile)
                        subprocess.call(["gzip", outfile])
                    if key == "mask":
                        resampled_lbl = preprocess_label(label=img_file,
                                                         target_spacing=target_spacing,
                                                         target_shape=target_shape,
                                                         target_orientation=('L', 'A', 'S'),
                                                         fill_value=0,
                                                         dtype=np.float32)
                        resampled_filename = patient + "_" + key + "_resampled" + ".nii"
                        outfile = new_pat_dir/resampled_filename
                        nib.save(resampled_lbl, outfile)
                        subprocess.call(["gzip", outfile])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc", help="zip-archive with nii.gz-images", required=True)
    parser.add_argument("--dst", help="output directory", required=True)

    args = parser.parse_args()
    archive = args.arc
    destination_dir = args.dst

    run_preprocessing(archive=archive, destination_dir=destination_dir)


if __name__ == '__main__':
    main()
