import h5py
from pathlib import Path
import numpy as np
import pandas as pd

single_dirs = [Path("/mnt/qdata/raheppt1/data/tumorvolume/interim/petct/TUE0000NORMA_2D.h5"),
               Path("/mnt/qdata/raheppt1/data/tumorvolume/interim/petct/TUE0001PETBC_2D.h5"),
               Path("/mnt/qdata/raheppt1/data/tumorvolume/interim/petct/TUE1001LYMPH_2D.h5"),
               Path("/mnt/qdata/raheppt1/data/tumorvolume/interim/petct/TUE1003MELPE_2D.h5")]

merge_into = Path("/mnt/qdata/raheppt1/data/tumorvolume/interim/petct/TUE0000ALLDS_2D.h5")
merge_into2 = Path("/mnt/qdata/raheppt1/data/tumorvolume/interim/petct/TUE0000ALLDS_3D_2.h5")

missing = [Path("/mnt/qdata/raheppt1/data/tumorvolume/interim/TUE0000NORMA2_3D.h5"),
           Path("/mnt/qdata/raheppt1/data/tumorvolume/interim/TUE0001PETBC_3D.h5"),
           Path("/mnt/qdata/raheppt1/data/tumorvolume/interim/TUE1001LYMPH_3D.h5"),
           Path("/mnt/qdata/raheppt1/data/tumorvolume/interim/TUE1003MELPE_3D.h5")]


def merge_3d(dirs_to_merge, merge_dir):
    error_log = {"dir": [], "key": [], "name": []}
    for src_dir in dirs_to_merge:
        project_name = src_dir.stem
        print(project_name)
        with h5py.File(src_dir, "r") as src:
            with h5py.File(merge_dir, "a") as dst:
                for key in src["/"].keys():
                    print(key)
                    dst.require_group(name=key)

                    def cp_it(name, dataset):
                        print(name)
                        try:
                            dst.copy(source=dataset,
                                     dest="/" + str(key) + "/" + str(name))
                        except Exception as e:
                            print(e)
                            error_log["dir"].append(src_dir)
                            error_log["key"].append(key)
                            error_log["name"].append(name)
                    src["/" + str(key)].visititems(cp_it)
    print(error_log)


def merge_2d(dirs_to_merge, merge_dir):
    array_dict = {"image": {}, "mask": {}}
    error_log = {"dir": [], "key": [], "name": []}
    for src_dir in dirs_to_merge:
        print(src_dir)
        with h5py.File(src_dir, "r") as src:
            for key in src["//data"].keys():
                print(key)
                array_dict[key][str(src_dir)] = src["//data/" + str(key)][:, :, :, -212:]

    concat_dict = {}
    for key in array_dict.keys():
        if key == "image":
            new_array = np.empty([0, 2, 200, 212])
        else:
            new_array = np.empty([0, 1, 200, 212])
        for prj, array in array_dict[key].items():
            print(array.shape)
            new_array = np.concatenate((new_array, array), axis=0)
        concat_dict[key] = new_array

    print(new_array.shape)
    print(concat_dict.keys())

    for key in concat_dict.keys():
        with h5py.File(merge_dir, "a") as dst:
            dst.require_dataset(name=key, data=concat_dict[key], shape=concat_dict[key].shape,
                                dtype=concat_dict[key].dtype)
    print(error_log)


merge_2d(single_dirs, merge_dir=merge_into)


def recover_missing_projects(missing_projects, merge_dir):
    error_log = {"dir": [], "key": [], "name": []}
    with h5py.File(merge_dir, "r") as src:
        for project in missing_projects:
            with h5py.File(project, "a") as dst:
                project_name = str(project.stem).split("_")[0]
                print(project_name)
                for key in ["image", "mask", "mask_iso"]:
                    print(key)
                    dst.require_group(name=key)

                    def copy_it(name, ds_object):
                        if ds_object.attrs["project"] == project_name:
                            print(name)
                            try:
                                dst.copy(source=ds_object,
                                         dest="/" + str(key) + "/" + str(name))
                            except:
                                error_log["dir"].append(project)
                                error_log["key"].append(key)
                                error_log["name"].append(name)
                    src["/" + key].visititems(copy_it)
    print(error_log)


def merge_merges(merge_dir, merge_dir2):
    error_log = {"dir": [], "key": [], "name": []}
    with h5py.File(merge_dir, "r") as src:
        with h5py.File(merge_dir2, "a") as dst:
            for key in ["image", "mask", "mask_iso"]:
                def copy_it(name, ds_object):
                    print(name)
                    try:
                        dst.copy(source=ds_object,
                                 dest="/" + str(key) + "/" + str(name))
                    except:
                        error_log["dir"].append(ds_object.attrs["project"])
                        error_log["key"].append(key)
                        error_log["name"].append(name)
                src["/" + key].visititems(copy_it)
    print(error_log)


def merge_csv(work_dir, out_file_name):
    work_dir = Path(work_dir)
    # print([pd.read_csv(f) for f in work_dir.glob("*.csv")])
    combined_csv = pd.concat([pd.read_csv(f) for f in work_dir.glob("*.csv")],
                             ignore_index=True,
                             join="inner")
    out_file = work_dir/out_file_name
    combined_csv.to_csv(str(out_file), encoding='utf-8-sig')


# merge_csv("/mnt/qdata/raheppt1/data/tumorvolume/interim/petct", "info_TUE0000ALLDS_2D.csv")

