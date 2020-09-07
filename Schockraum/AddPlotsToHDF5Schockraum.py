from pathlib import Path
import h5py
import zarr
import scipy.ndimage
import numpy as np
import pandas as pd
import argparse
from joblib import Parallel, delayed


def add_plots_to_hdf5(source_file, destination_file, lo, hi, step, jobs):
    """
    Adds for each given angle a concatenated array to the hdf5.
    Args:
        source_file (Path): A Path-object representing the path to the hdf5-file where the data is stored.
        destination_file (Path): A Path-object representing the path to the hdf5-file the 2D-arrays are stored to.
        lo (int): The starting angle.
        hi (int): The last angle (not included).
        step (int): The step between angles.
        jobs (int): Number of parallel jobs.

    Returns:
        None
    """
    with h5py.File(source_file, "r") as hf1:

        jobs = int(jobs)
        lo = int(lo)
        hi = int(hi)
        step = int(step)
        angles = np.arange(lo, hi, step)
        z_min = 100000
        slice_number = 0
        slice_k_angle_list = []
        info_dict = {"number": [], "patient": [], "angle": [], "project": []}
        error_dict = {"number": [], "patient": [], "angle": [], "project": []}

        for k in hf1["image"]:
            pat_to_test = hf1[f"image/{k}"]
            for angle in angles:
                # arrays are shaped (0, xy_shape, xy_shape, z) by RunPreprocessing.py
                if pat_to_test.shape[3] >= 200:
                    slice_number += 1
                    slice_k_angle_list.append((slice_number, k, angle))
                    info_dict["number"].append(slice_number - 1)
                    info_dict["patient"].append(k)
                    info_dict["angle"].append(angle)
                    info_dict["project"].append(hf1[f"image/{k}"].attrs["project"])
                    if pat_to_test.shape[3] <= z_min:
                        z_min = pat_to_test.shape[3]
                        xy_shape = pat_to_test.shape[1]
                else:
                    error_dict["number"].append(slice_number - 1)
                    error_dict["patient"].append(k)
                    error_dict["angle"].append(angle)
                    error_dict["project"].append(hf1[f"image/{k}"].attrs["project"])

        # save info_dict as pd-DataFrame in a .csv
        destination_file = Path(destination_file)
        df = pd.DataFrame.from_dict(info_dict)
        project_name = destination_file.stem
        out_dir = destination_file.parent
        print(out_dir, project_name)
        filename = "info_" + project_name + ".csv"
        outfile = out_dir / filename
        df.to_csv(outfile)

        # save error_dict as pd-DataFrame in a .csv
        error_df = pd.DataFrame.from_dict(error_dict)
        out_dir = destination_file.parent
        error_filename = "error_" + project_name + ".csv"
        error_outfile = out_dir / error_filename
        error_df.to_csv(error_outfile)

        image_array_shape = (slice_number, 1, xy_shape, z_min)
        print(image_array_shape)

        def proc(i, k, angle):
            with h5py.File(source_file, "r") as hf1:
                print(i, k, angle)
                image = np.array(hf1[f"image/{k}"][:, :, :, -z_min:]).astype(np.float32)
                if not angle == 0 or angle == 360:
                    ct = scipy.ndimage.rotate(image[0, :, :, :], angle=angle, reshape=False).astype(np.float16)
                    new_image = ct[len(ct) // 2, :, :]
                    new_image = new_image[np.newaxis, :, :]
                else:
                    ct = np.array(image[0, :, :, :]).astype(np.float16)
                    new_image = ct[len(ct) // 2, :, :]
                    new_image = new_image[np.newaxis, :, :]

            return i, new_image

        def chunky(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                if not (i + n) > len(lst):
                    so_chunky = lst[i:i + n]
                else:
                    so_chunky = lst[i:len(lst)]
                yield so_chunky

        # break down patients ("k") into chunks for multiprocessing
        k_chunks = list(chunky(slice_k_angle_list, jobs))

    z_mem = zarr.group("data")
    z_image = z_mem.create_dataset("image", shape=image_array_shape, dtype=np.float16, overwrite=True)

    for chunk in k_chunks:
        results = Parallel(n_jobs=min(jobs, len(chunk)))(delayed(proc)(i=i - 1, k=k, angle=angle)
                                                         for i, k, angle in chunk)

        for i, nu_image in results:
            z_image[i, :, :] = nu_image

    with h5py.File(destination_file, "a") as hf2:
        data = hf2.require_group(name="data")
        zarr.convenience.copy_all(source=z_mem, dest=data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="source hdf5-directory", required=True)
    parser.add_argument("--dst", help="destination hdf5-directory", required=True)
    parser.add_argument("--lo", help="start of rotation", default=0)
    parser.add_argument("--hi", help="end of rotation", default=91)
    parser.add_argument("--stp", help="step rotation", default=90)
    parser.add_argument("--job", help="number of parallel workers", default=10)

    args = parser.parse_args()

    add_plots_to_hdf5(source_file=args.src,
                      destination_file=args.dst,
                      lo=args.lo,
                      hi=args.hi,
                      step=args.stp,
                      jobs=args.job)


if __name__ == '__main__':
    main()
