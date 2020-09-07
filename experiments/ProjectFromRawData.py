import pandas as pd
from pathlib import Path
import shutil
from hashlib import sha3_256
import glob
import re


class ProjectFromRawData:
    def __init__(self, workdir, targetdir):
        self.workdir = Path(workdir)
        self.targetdir = Path(targetdir)
        self.projectname = str(self.workdir).split("/")[-1]
        self.find_and_write()

    def find_and_write(self):
        projectdir = self.targetdir/self.projectname
        try:
            projectdir.mkdir()
        except FileExistsError:
            pass

        name_list = []
        hash_list = []
        requirements_list = []
        for patient in self.workdir.iterdir():
            for sequence in patient.iterdir():

                pat_name = str(patient).split("/")[-1]
                print(pat_name)
                pat_hash = self.name_to_hash(patient, sequence)
                print(pat_hash)

                ct_exists = glob.glob(str(sequence/Path("**/*GK*.nii")), recursive=True)
                pet_exists = glob.glob(str(sequence/Path("**/PET*.nii")), recursive=True)
                mask_exists = glob.glob(str(sequence/Path("**/*mask*.nii*")), recursive=True)

                if not ct_exists == list() and not pet_exists == list() and not mask_exists == list():
                    patdir = projectdir/str(pat_hash)
                    patdirseq = patdir/"sequences"
                    patdirlbl = patdir/"labels"
                    try:
                        patdir.mkdir()
                        patdirseq.mkdir()
                        patdirlbl.mkdir()
                    except FileExistsError:
                        pass

                    shutil.copy(Path(ct_exists[0]), patdirseq)
                    shutil.copy(Path(pet_exists[0]), patdirseq)
                    shutil.copy(Path(mask_exists[0]), patdirlbl)

                    r = re.compile("s\d{3}")
                    pet_seq = r.findall(pet_exists[0])[0]
                    dicom = glob.glob(str(sequence / Path("**/*" + pet_seq + "*.dcm")), recursive=True)
                    if not dicom == list():
                        missing = "nothing missing"
                        shutil.copy(dicom[0], patdirseq)
                    else:
                        missing= "Dicom missing"

                    name_list.append(pat_name)
                    hash_list.append(pat_hash)
                    requirements_list.append(missing)

                else:
                    missing = "missing: "
                    if ct_exists == list():
                        missing += " ct"
                    else:
                        pass
                    if pet_exists == list():
                        missing += " pet"
                    else:
                        pass
                    if mask_exists == list():
                        missing += " mask"
                    else:
                        pass
                    name_list.append(pat_name)
                    hash_list.append(pat_hash)
                    requirements_list.append(missing)
        df = pd.DataFrame({"patient name": pd.Series(name_list),
                           "patient hash": pd.Series(hash_list),
                           "requirements.txt": pd.Series(requirements_list)})
        df.to_csv(path_or_buf=projectdir/Path("patientinfo.csv"))


    def name_to_hash(self, patient, sequence):
        name = str(patient).split("/")[-1]
        seq = str(sequence).split("/")[-1]
        hash_name_long = sha3_256(str(name).encode("utf-8")).hexdigest()
        hash_seq_long = sha3_256(str(seq).encode("utf-8")).hexdigest()
        hash_name = hash_name_long[:5]
        seq_name = hash_seq_long[:5]
        complete_hash = hash_name + seq_name
        return Path(complete_hash)


just_do_it = ProjectFromRawData("/media/nora/imgdata/TUE1003MELPE/",
                                "/media/ssd/rakerbb1/TumorVolume")
