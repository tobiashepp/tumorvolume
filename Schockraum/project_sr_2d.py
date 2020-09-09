import numpy as np
import nibabel as nib
from pathlib import Path
from nilearn.image import resample_img
import matplotlib.pyplot as plt
import h5py
from joblib import Parallel, delayed

work_dir = Path("/mnt/qdata/rakerbb1/data/Schockraum/raw/cerebrum/")
# get all dirs except for custom atlases
dir_numbers = np.arange(0, len(list(work_dir.iterdir())) - 3, 1)
dirs_to_draw = []
i = -1
for study_dir in work_dir.iterdir():
    i += 1
    if i in dir_numbers:
        dirs_to_draw.append(str(study_dir))

print("total: " + str(len(list(work_dir.iterdir())) - 3) + "\n" + "first: " + dirs_to_draw[0] + "\n" + "last: " +
      dirs_to_draw[-1])

# Preprocessing for 2D
jobs = 20
shape = 224

dir_nums = np.arange(0, len(dirs_to_draw))
i_dirs = list(zip(dir_nums, dirs_to_draw))

cerebrum_arr = np.zeros((len(dirs_to_draw), 3, shape, shape))

error_dict = {"layer": [], "pat_hash": [], "cause": []}


def proc(tpl):
    i = tpl[0]
    study_dir = Path(tpl[1])
    pat_hash = study_dir.name

    file_name = pat_hash + "_ants_reg_strip.nii"
    try:
        img = nib.load(study_dir/file_name)

        res_img = resample_img(img,
                               target_affine=img.affine,
                               target_shape=(shape, shape, shape),
                               interpolation="continuous",
                               fill_value=0)
        res_arr = res_img.get_fdata().astype(np.float32)

        arr_min = 0
        arr_max = np.max(res_arr)
        res_arr -= arr_min
        res_arr *= (1 / (arr_max - arr_min))
        res_arr = np.clip(res_arr, 0, 1)
        x_arr = res_arr[res_arr.shape[0] // 2, :, :]
        y_arr = res_arr[:, res_arr.shape[1] // 2, :]
        z_arr = res_arr[:, :, res_arr.shape[2] // 2]
        new_arr = np.stack((x_arr, y_arr, z_arr), axis=0)

        fig, axs = plt.subplots(1, 3)
        fig.suptitle(pat_hash)
        axs[0].imshow(np.rot90(x_arr), cmap="tab20b")
        axs[1].imshow(np.rot90(y_arr), cmap="tab20b")
        axs[2].imshow(np.rot90(z_arr), cmap="tab20b")
        img_dir = Path("/mnt/qdata/rakerbb1/data/Schockraum/raw/dl_images/")
        img_dir.mkdir(exist_ok=True)

        out_file_name = pat_hash + "_arr_dl.png"
        out_file = img_dir/out_file_name
        plt.savefig(out_file, bbox_inches="tight")

        plt.close(fig)

        print(pat_hash, new_arr.shape)

        return i, new_arr

    except Exception as e:
        print(e)
        error_dict["layer"].append(i)
        error_dict["pat_hash"].append(pat_hash)
        error_dict["cause"].append(e)
        return i, np.zeros((3, shape, shape))


def chunky(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        if not (i + n) > len(lst):
            so_chunky = lst[i:i + n]
        else:
            so_chunky = lst[i:len(lst)]
        yield so_chunky


# break down patients ("k") into chunks for multiprocessing
k_chunks = list(chunky(i_dirs, jobs))

chunk_i = 0
num_chunks = len(dirs_to_draw) // jobs

for chunk in k_chunks:
    chunk_i = chunk_i + 1
    print("chunk " + str(chunk_i) + " of " + str(num_chunks + 1))
    results = Parallel(n_jobs=min(jobs, len(chunk)))(delayed(proc)(tpl=tpl) for tpl in chunk)

    for result in results:
        cerebrum_arr[result[0]] = result[1]

h5_file = "/mnt/qdata/rakerbb1/data/Schockraum/interim/cerebrum/cerebrum_2d.h5"

with h5py.File(h5_file, "a") as hf:
    cerebrum = hf.require_dataset(name="cerebrum",
                                  data=cerebrum_arr,
                                  shape=cerebrum_arr.shape,
                                  dtype=cerebrum_arr.dtype)
print(error_dict)
