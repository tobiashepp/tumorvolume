from pathlib import Path
import pandas as pd
import re
import shutil
import subprocess
from midastools.pet.corr2suv import Pet
import SimpleITK as Sitk
from hashlib import sha3_256

work_dir = Path("/media/nora/imgdata/...")
nora_dir = Path("/media/nora/work")

data_dict = {"patient": [], "study": [], "hash": []}
new_project_dir = nora_dir / work_dir.name
new_project_dir.mkdir(exist_ok=True)

csv_filename = str(work_dir.name) + "_pat_data_mask.csv"
out_file_csv = nora_dir / Path("anonym") / csv_filename

log_path = "/media/ssd/rakerbb1/test.txt"


def process_study(study_dir, log_file, study_hash, out_dir):
    try:
        # find matching files for pet, ct and mask. The keyword petsuv initially stores a pet but is later
        # used to name the processed petsuv-file.
        files = {"petsuv": None, "ct": None, "mask": None, "dcm": None}
        pattern = {"petsuv": "**/PET_corr*.nii",
                   "mask": "**/*mask*",
                   "ct": "**/*GK*.nii*"}
        for key in ["petsuv", "ct", "mask"]:
            files[key] = next(study_dir.glob(pattern[key]))

        # pat_dir has to be created here to make sure, that an error is raised if
        pat_dir = out_dir / study_hash
        pat_dir.mkdir(exist_ok=True)

        for key in ["petsuv", "ct", "mask"]:
            # find file endings for files
            suffixes = files["key"].suffixes
            if len(suffixes) > 1:
                suffix = suffixes[0] + suffixes[1]
            else:
                suffix = suffixes[0]

            new_filename = study_hash + "_" + key + suffix
            out_file = str(pat_dir / new_filename)

            if key == "pet":
                # find DICOM-header matching pet-sequence
                suffix = re.match(".*(s\d{3})", files["petsuv"].stem)[1]
                files["dcm"] = next(study_dir.glob(f"**/*{suffix}.dcm"))
                # calculate petsuv from pet and dcm
                pet = Pet(nii_path=files["pet"], dcm_header_path=files["dcm"])
                suvpet = pet.calc_suv_image()
                suvpet32 = Sitk.Cast(suvpet, Sitk.sitkFloat32)
                Sitk.WriteImage(suvpet32, out_file)
            else:
                shutil.copy(files[key], out_file)

            if not files[key].suffix == ".gz":
                subprocess.call(["gzip", out_file])
        return True

    except:
        print("error")
        log_file.write(f"error {study_dir} ")
        for key, file in files.items():
            if not file:
                log_file.write(f" missing {key}")
                print(f" missing {key}")
        log_file.write("\n")
        return False


def name_to_hash(patient, study):
    hash_name_long = sha3_256(patient.encode("utf-8")).hexdigest()
    hash_seq_long = sha3_256(study.encode("utf-8")).hexdigest()
    hash_name = hash_name_long[:7]
    seq_name = hash_seq_long[:3]
    complete_hash = hash_name + seq_name
    return complete_hash


with open(log_path, "w") as log_file:
    for pat in work_dir.iterdir():
        for study in pat.iterdir():
            study_hash = name_to_hash(pat.name, study.name)
            print(pat.name, study.name)
            if process_study(study, log_file, study_hash, new_project_dir):
                data_dict["patient"].append(pat.name)
                data_dict["study"].append(study.name)
                data_dict["hash"].append(study_hash)

df = pd.DataFrame.from_dict(data_dict)
df.to_csv(out_file_csv)

subprocess.call(["cat", "/home/rakerbb1/test.txt"])
