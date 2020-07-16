import h5py
import pandas as pd
from pathlib import Path
from tqdm as tqmd
import numpy as np
from pathlib import Path
from scipy.ndimage import label

### PET CT
# Validation of the lesion predictions for the test data 
# (with ground truth labels as reference).

# paths
work_dir = Path('/mnt/qdata/raheppt1/data/tumorvolume/')
data_path = work_dir/'interim/petct/TUE0000ALLDS_3D.h5'

# get key list
with h5py.File(data_path, 'r') as hf:
    keys = list(hf['image'])

result_dict = {'key': [], 
            'project': [],
            'volume_tumor': [],
            'uptake_tumor': [],
            'uptake_tumor_mean': [],
            'uptake_tumor_std': [],
            'uptake_tumor_median': [],
            'uptake_tumor_min': [],
            'uptake_tumor_max': [],
            'num_lesions': []}

for key in tqdm(keys):
    print(key)
    # load training data
    with h5py.File(data_path, 'r') as hf:
        ds = hf[f'image/{key}']
        img = ds[0,:].astype(np.float32)
        project = ds.attrs['project']
        affine = ds.attrs['affine'] 
        mask = hf[f'mask_iso/{key}'][0, :].astype(int)

    # volumina
    voxel_vol = 2*2*3 # mm^3
    petsuv_scale = 40
    volume_tumor = np.sum(mask)*voxel_vol
    uptake = np.ma.array(mask*img, mask=mask==0)*petsuv_scale
    _, num_lesions = label(mask)
    # number of lesions 

    # save  results
    result_dict['key'].append(key)
    result_dict['project'].append(project)
    result_dict['volume_tumor'].append(volume_tumor)
    result_dict['uptake_tumor'].append(uptake.sum())
    result_dict['uptake_tumor_mean'].append(uptake.mean())
    result_dict['uptake_tumor_std'].append(uptake.std())
    result_dict['uptake_tumor_median'].append(np.ma.median(uptake.mean()))
    result_dict['uptake_tumor_min'].append(uptake.min())
    result_dict['uptake_tumor_max'].append(uptake.max())
    result_dict['num_lesions'].append(num_lesions)

# csv results to csv
df = pd.DataFrame.from_dict(result_dict)
df.to_csv(work_dir/'processed/petct/petct_analysis.csv')