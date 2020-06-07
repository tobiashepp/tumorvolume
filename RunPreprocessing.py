from midasvessel.processing.preprocessing import preprocess_image, preprocess_label
import re
import nibabel as nib
import subprocess
import numpy as np
from io import BytesIO
from zipfile import ZipFile
import gzip
from pathlib import Path

raw_dir = Path("/mnt/qdata/raheppt1/data/tumorvolume/raw")

project_archives = ["TUE1001LYMPH.zip"]  # str
# ["TUE0001PETBC.zip", "TUE1001LYMPH.zip", "TUE1003MELPE.zip"]

interim_dir = Path("/mnt/qdata/raheppt1/data/tumorvolume/interim")


# the ziparchive used was created on another disk thus saving the old filepaths.
# We need to change the filepath to store the resampled images.


for archive in project_archives:
    file = raw_dir/archive
    project_name = file.stem
    new_project_dir = interim_dir/project_name
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
            for file in files[key][:1]:
                file_path = Path(file)
                item = file_path.name
                if key == "petsuv":
                    patient = str(file_path.parent).split("/")[-1]
                    print(patient)
                    new_pat_dir = interim_dir/patient
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
                        subprocess.call(["gzip", outfile])  #1
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
