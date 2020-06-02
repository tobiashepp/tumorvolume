from pathlib import Path
from midastools.pet.corr2suv import Pet
import SimpleITK as sitk

work_dir = Path("/media/nora/work/TUE1001LYMPH")
for file in work_dir.glob('**/*pet.nii'):
    print(file)
    dcm_header = file.parent/file.name.replace('pet.nii', 'dcm.dcm')
    print(dcm_header)
# for patient in work_dir.glob("**"):
#     pet_seq = patient.glob("/*pet.nii")
#     print(pet_seq)
#     hdr = patient.glob("*dcm.dcm")
#     #calc_suv = Pet(nii_path=pet_seq, dcm_header_path=hdr)
#     suvPET = calc_suv.calc_suv_image()
#     if pet_seq:
#         filename = str(patient.name + "_suvpet.nii")
#         save_at = patient/filename
#         print(save_at)
#         img = sitk.ReadImage(nii_file)
#         print('size', img.GetSize())
#         print(nii_file, dcm_header, out_file)
#         pet_obj = Pet(nii_path=nii_file, dcm_header_path=dcm_header)
#         image_pet_suv = pet_obj.calc_suv_image().astype(args.dtype)
#         sitk.WriteImage(image_pet_suv, out_file)