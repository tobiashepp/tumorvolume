import h5py
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import gzip
import re
import numpy as np
import nibabel as nib
from sklearn.model_selection import KFold
from midastools.misc.orientation import reorient_nii

work_dir = Path("/mnt/qdata/raheppt1/data/tumorvolume/")
zip_path = work_dir/"raw/TUE0000CTORG_pp.zip"
h5_path = work_dir/"interim/ctorgans.h5"
ct_range = [-1000.0, 1000.0]
target_orientation = ("L","A","S")

# extract nifti data from zip file, reorient, normalize 
# and write to hdf5
# nifti data is assumed to be resampled to (2,2,3)mm and must be LAS compatibel
with h5py.File(h5_path, "w") as hf:
    with ZipFile(zip_path, "r") as zf:
        # parse filelist
        file_list = zf.namelist()
        label_files = [f for f in file_list if re.match(".*label.*.nii.gz", f)]
        img_files = [f for f in file_list if re.match(".*ct.*.nii.gz", f)]

        for flabel, fimg in zip(label_files, img_files):
            # get keys
            print(flabel, fimg)
            key_label = re.match(".*/(\d{3}).*nii.gz", flabel)[1]
            key_img   = re.match(".*/(\d{3}).*nii.gz", flabel)[1]
            assert key_img == key_label

            print(key_img, key_label)
            #read nifti data from zip
            with zf.open(fimg, "r") as zfile:
                gzfile = gzip.GzipFile(fileobj=BytesIO(zfile.read()), mode="rb")
                fh = nib.FileHolder(fileobj=BytesIO(gzfile.read()))
                img = nib.Nifti1Image.from_file_map({"header": fh, "image": fh, "affine": fh})

            with zf.open(flabel, "r") as zfile:
                gzfile = gzip.GzipFile(fileobj=BytesIO(zfile.read()), mode="rb")
                fh = nib.FileHolder(fileobj=BytesIO(gzfile.read()))
                label = nib.Nifti1Image.from_file_map({"header": fh, "image": fh, "affine": fh})

            print(f"img  : {img.shape}, {img.header.get_zooms()}, {nib.aff2axcodes(img.affine)}")
            print(f"label: {label.shape}, {label.header.get_zooms()}, {nib.aff2axcodes(label.affine)}")

            # reorient niftis 
            img = reorient_nii(img, target_orientation)
            label = reorient_nii(label, target_orientation)

            # normalize ct data
            img_array = img.get_fdata()
            img_array = (img_array - ct_range[0])/(ct_range[1] - ct_range[0])
            img_array = np.clip(img_array, 0, 1)
            img_array = np.expand_dims(img_array, 0).astype(np.float16)

            label_array = label.get_fdata()
            label_array = np.expand_dims(label_array, 0).astype(np.uint8)

            # store to hdf5
            gr = hf.require_group("images")
            ds = gr.require_dataset(key_img, img_array.shape, img_array.dtype)
            ds[:] = img_array

            gr = hf.require_group("labels")
            ds = gr.require_dataset(key_label, label_array.shape, label_array.dtype)
            ds[:] = label_array


# generate key files for cross validation
with h5py.File(h5_path, "r") as hf:
    subject_keys = list(hf["images"])
kf = KFold(n_splits=5)
for k, (train_index, test_index) in enumerate(kf.split(subject_keys)):
    train_keys = [(subject_keys[i] + "\n") for i in train_index]
    with Path(h5_path.parent/(h5_path.stem+f"_train{k}.dat")).open("w") as keyfile:
        keyfile.writelines(train_keys)
    test_keys = [(subject_keys[i] + "\n") for i in test_index]
    with Path(h5_path.parent/(h5_path.stem+f"_test{k}.dat")).open("w") as keyfile:
        keyfile.writelines(test_keys)
