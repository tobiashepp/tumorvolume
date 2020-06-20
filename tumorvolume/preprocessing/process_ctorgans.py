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
h5_path = work_dir/"interim/ctorgans/ctorgans.h5"
ct_range = [-1000.0, 1000.0]
target_orientation = ("L","A","S")

# extract nifti data from zip file, reorient, normalize 
# and write to hdf5
# nifti data is assumed to be resampled to (2,2,3)mm and must be LAS compatibel
# CAVE: There is a orientation bug in some of the zipped nifti masks!!!
# LPI should be LPS!!! Therefore, the affine matrix is manually corrected.
with h5py.File(h5_path, "w") as hf:
    with ZipFile(zip_path, "r") as zf:
        # parse filelist
        file_list = zf.namelist()
        label_files = [f for f in file_list if re.match(".*labels_3mm.*.nii.gz", f)]
        img_files = [f for f in file_list if re.match(".*ct_3mm.*.nii.gz", f)]

        for flabel, fimg in zip(label_files, img_files):
            # get keys
            print(flabel, fimg)
            key_label = re.match(".*/(\d{3}).*nii.gz", fimg)[1]
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

            if nib.aff2axcodes(img.affine)[2] == "I":
                print("fixing orientation bug {nib.aff2axcodes(img.affine)}!") 
                # Bug fixing: LPI -> LPS for corrupted images
                img.affine[:, 2] = -img.affine[:, 2]

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
            ds.attrs["affine"] = img.affine.tolist()

            gr = hf.require_group("labels")
            ds = gr.require_dataset(key_label, label_array.shape, label_array.dtype)
            ds[:] = label_array
            ds.attrs["affine"] = label.affine.tolist()


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

import matplotlib.pyplot as plt
validation_dir = (h5_path.parent/f"validation")
validation_dir.mkdir(exist_ok=True)
with h5py.File(h5_path, "r") as hf:
    subject_keys = list(hf["images"])
    for k in subject_keys:
        print(k)
        img = hf[f"images/{k}"][:].astype(np.float32)
        mask = hf[f"labels/{k}"][:].astype(np.float32)
        fig, ax = plt.subplots()
        ax.imshow(img[0].max(axis=1), cmap="bone")
        ax.imshow(mask[0].max(axis=1), cmap="coolwarm", alpha=0.5)
        plt.savefig(str(validation_dir/f"validation{k}.png"))
        plt.close(fig)