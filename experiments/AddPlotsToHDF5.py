from pathlib import Path
import h5py
import scipy.ndimage
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def add_plots_to_hdf5(source_file, destination_file, angles, jobs):
    """
    Adds for each given angle a concatenated array consisting of a CT, a PET, a Mask and an Isocontour-Mask to the hdf5.
    Args:
        source_file (Path): A Path-object representing the path to the hdf5-file where the data is stored.
        destination_file (Path): A Path-object representing the path to the hdf5-file the 2D-arrays are stored to.
        angles (list of int): A list of ints representing angles for the arrays to be rotated by.
        jobs (int): Number of parallel jobs.

    Returns:
        None
    """
    with h5py.File(source_file, "r") as hf1:
        z_min = 100000
        for k in hf1["image"]:
            pat_to_test = hf1[f"image/{k}"]
            if pat_to_test.shape[3] < z_min:
                z_min = pat_to_test.shape[3]

        def proc(k):
            with h5py.File(source_file, "r") as hf1:
                print(k)
                inside_angle_dict = {}

                image = np.array(hf1[f"image/{k}"][:, :, :, -z_min:]).astype(np.float32)
                mask = np.array(hf1[f"mask/{k}"][:, :, :, -z_min:]).astype(np.float32)
                mask_iso = np.array(hf1[f"mask_iso/{k}"][:, :, :, -z_min:]).astype(np.float32)

                for inside_angle in angles:
                    print(inside_angle)
                    if not inside_angle == 0 or inside_angle == 360:
                        pet = np.max(scipy.ndimage.rotate(image[0, :, :, :], angle=inside_angle, reshape=False),
                                     axis=1).astype(np.float16)
                        ct = np.max(scipy.ndimage.rotate(image[1, :, :, :], angle=inside_angle, reshape=False),
                                    axis=1).astype(np.float16)
                        new_image = np.stack([pet, ct], axis=0)
                        new_mask_iso = np.max(scipy.ndimage.rotate(mask_iso[0, :, :, :], angle=inside_angle, reshape=False),
                                              axis=1).astype(np.uint8)
                    else:
                        pet = np.max(image[0, :, :, :], axis=1).astype(np.float16)
                        ct = np.max(image[1, :, :, :], axis=1).astype(np.float16)
                        new_image = np.stack([pet, ct])
                        new_mask_iso = np.max(mask_iso[0, :, :, :], axis=1).astype(np.uint8)

                    inside_arrays = {"image": new_image,
                                     "mask_iso": new_mask_iso,
                                     "angle": inside_angle,
                                     "project": hf1[f"mask_iso/{k}"].attrs["project"]}
                    inside_angle_dict[inside_angle] = inside_arrays
            return k, inside_angle_dict

        def chunky(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), jobs):
                if not (i + n) > len(lst):
                    so_chunky = lst[i:i + jobs]
                else:
                    so_chunky = lst[i:len(lst)]
                yield so_chunky

        k_chunks = list(chunky(list(hf1["image"]), 5))
    i = 0

    info_dict = {"number": [], "patient": [], "angle": [], "project": []}
    image_arr = None
    mask_arr = None
    for chunk in k_chunks:
        k_dict_tuple_list = Parallel(n_jobs=min(jobs, len(chunk)))(delayed(proc)(k=key) for key in chunk)

        with h5py.File(destination_file, "a") as hf2:
            for k_dict_tuple in k_dict_tuple_list:
                k = k_dict_tuple[0]
                angle_dict = k_dict_tuple[1]
                print(k)
                for angle in angle_dict.keys():
                    i += 1
                    group = hf2.require_group(str(i))
                    arrays = angle_dict[angle]

                    info_dict["number"].append(i)
                    info_dict["patient"].append(k)
                    info_dict["angle"].append(angle)
                    info_dict["project"].append(angle_dict[angle]["project"])
                    for item in ["image", "mask_iso"]:
                        print(item)
                        array = arrays[item]
                        group.attrs["patient"] = k
                        group.attrs["project"] = angle_dict[angle]["project"]
                        group.require_dataset(name=item, data=array, shape=array.shape, dtype=array.dtype)

    df = pd.DataFrame.from_dict(info_dict)
    project_name = destination_file.stem
    out_dir = destination_file.parent
    filename = "info_" + project_name + ".csv"
    outfile = out_dir/filename
    df.to_csv(outfile)


src_file = Path("/mnt/qdata/raheppt1/data/tumorvolume/interim/TUE1001LYMPH_3D.hdf5")
dst_file = Path("/mnt/qdata/raheppt1/data/tumorvolume/interim/TUE1001LYMPH_2D.h5")
angls = [0, 20, 40, 60, 80]
add_plots_to_hdf5(source_file=src_file, destination_file=dst_file, angles=angls, jobs=5)
