import zipfile
import nibabel as nib
import re
import pandas as pd
from io import BytesIO

class ReadFromZip:
    def __init__(self, zip_filepath):
        self.zip_file = zipfile.ZipFile(zip_filepath)
        self.complete_list = self.zip_file.namelist()
        print(self.complete_list)
        #self.study_info()
        #self.zip_to_nii()


    def study_info(self):
        r1 = re.compile("\d{3}/")
        pre_study_list = list(filter(r1.fullmatch, self.complete_list))
        study_list = []
        ct_list = []
        petsuv_list = []
        mask_list = []
        missing_list = []
        for study in pre_study_list:
            r_study =re.compile(study)
            study_files_list = list(filter(r_study.match, self.complete_list))
            r_ct = re.compile(".*ct.nii.*")
            r_pet = re.compile(".*petsuv.nii.*")
            r_mask = re.compile(".*tumor.nii.*")
            ct = list(filter(r_ct.match, study_files_list))
            petsuv = list(filter(r_pet.match, study_files_list))
            mask = list(filter(r_mask.match, study_files_list))
            if not ct ==list() and not petsuv == list() and not mask == list():
                study_list.append(study)
                ct_list.append(str(ct[0]))
                petsuv_list.append(str(petsuv[0]))
                mask_list.append((str(mask[0])))
            else:
                if ct == list():
                    missing_list.append(study + ": ct")
                else:
                    pass
                if petsuv == list():
                    missing_list.append(study + ": pet")
                else:
                    pass
                if mask == list():
                    missing_list.append(study + ": mask")
                else:
                    pass

        df = pd.DataFrame({"studies" : pd.Series(study_list),
                           "ct" : pd.Series(ct_list),
                           "pet" : pd.Series(petsuv_list),
                           "mask" : pd.Series(mask_list),
                           "missing" : pd.Series(missing_list)})
        print(df)

    def zip_to_nii(self):
        r = re.compile(".*.nii")
        nii_list = list(filter(r.match, self.complete_list))
        print(nii_list)
        fh = nib.FileHolder(fileobj=BytesIO(self.zip_file.read(nii_list[1])))
        img_file = nib.Nifti1Image.from_file_map({"header":fh, "image": fh})
        return img_file


just_do_it = ReadFromZip("/media/nora/TUE1001LYMPH/")

