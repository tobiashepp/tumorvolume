import h5py
from pathlib import Path

single_dirs = [Path("/mnt/qdata/raheppt1/data/tumorvolume/interim/TUE1001LYMPH_3D.h5"),
               Path("/mnt/qdata/raheppt1/data/tumorvolume/interim/TUE1003MELPE_3D.h5"),
               Path("/mnt/qdata/raheppt1/data/tumorvolume/interim/TUE0001PETBC_3D.h5")]

merge_dir = Path("/mnt/qdata/raheppt1/data/tumorvolume/interim/TUE0001_1001_1003_3D.h5")

for src_dir in single_dirs:
    with h5py.File(src_dir, "r") as src:
        with h5py.File(merge_dir, "a") as dst:
            for key in src["/"].keys():
                dst_grp = dst.require_group(name=key)
                for ds in src["/" + str(key)].visititems():
                    src.copy(source=ds, dest=dst)
