from pathlib import Path
from midastools.pet.corr2suv import Pet
import SimpleITK as sitk

work_dir = Path("/media/nora/work/TUE1003MELPE")
for file in work_dir.glob('**/*pet.nii'):
    pet_seq = str(file)
    dcm_hdr = str(file.parent/file.name.replace('pet.nii', 'dcm.dcm'))
    pet_obj = Pet(nii_path=pet_seq, dcm_header_path=dcm_hdr)
    suvPET = pet_obj.calc_suv_image()
    suvPET32 = sitk.Cast(suvPET, sitk.sitkFloat32)
    new_filename= file.parent.name + "_petsuv.nii"
    out_file = str(file.parent/new_filename)
    print(pet_seq, dcm_hdr, out_file)
    sitk.WriteImage(suvPET32, out_file)