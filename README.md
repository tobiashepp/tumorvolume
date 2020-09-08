# tumorvolume
Our raw patient data consisted of multiple projects which had the following structure in common: 
project
       \ 
         patient
         patient
                \
                 study
                 study
                      \
                        CT
                        PET
                        MASK
                        DICOM-header
                        
We used createproject.py to find the patients that fulfilled the requirements of having a CT- and PET-sequence as well as a labeled sequence (MASK) and a DICOM-header. 
The patients names and studies were anonymized using sha256-hashes, the first 7 letters/digits from the name, the last 3 from the study. The radiological information of the DICOM-header were used to calculate the PETSUV of the PET-sequences. 
The sequences where renamed to hash + (ct, pet, mask) + nii.gz and zipped into a single file. The new structures wss:
zip(project
           \ 
             hash
             hash
                    \
                      hash_ct.nii.gz
                      hash_petsuv.nii.gz
                      hash_mask.nii.gz)
preprocessing.py is executable via the command line. It uses multiprocessing to read a given zip-file, preprocess the contents and store the results in a hdf5-file. 
First, the images are reoriented to "L", "A", "S"-Orientation. The CT is resampled to the desired shape and spacing using nilearn.resample_img. The other images are resampled to 
the CT using nilearn.resample_to_img. For the MASK, an isocontour is calculated. Each image array is normalized and stored in the hdf5 as a dataset (name=hash) in one of the 
three groups of the hdf5:
image = [2, xy-shape, xy-shape, z] where [0, :, :, :] is the PET and [1, :, :, :] is the CT.
mask = [1, xy-shape, xy-shape, z]
mask_iso = [1, xy-shape, xy-shape, z]

The project-hdf5-files can be merged with mergeh5.py.

project_2d.py adds 2D-plots to another hdf5. The array is read from the 3D-hdf5, the desired rotations are applied (for data augmentation purposes) and with np.max a 2D array created. Each slice is stored in a single dataset in the 2D-hdf5. project_2d.py is executable via the commandline and also capable of multiprocessing. 

The hdf5-files are now ready to be used for the learning task. 
