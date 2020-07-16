import zarr
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from midasmednet.unet.loss import compute_per_channel_dice, expand_as_one_hot
import torch
import pandas as pd
import nibabel as nib
from pathlib import Path
import numpy as np
import nibabel as nib
from pathlib import Path
from tumorvolume.ctorgans.postprocessing import largest_component

### PET CT
# Validation of the lesion predictions for the test data 
# (with ground truth labels as reference).

# paths
work_dir = Path('/mnt/qdata/raheppt1/data/tumorvolume/')
data_path = work_dir/'interim/petct/TUE0000ALLDS_3D.h5'
prediction_path = work_dir/'processed/petct/petct_validation.zip'
png_dir = Path(work_dir/'processed/petct/petct_validation_png' )
nifti_dir =  work_dir/'processed/petct/petct_validation_nifti'
png_dir.mkdir(exist_ok=True)
nifti_dir.mkdir(exist_ok=True)

# get key list
with zarr.open(store=zarr.ZipStore(prediction_path, mode='r')) as zf:
    keys = list(zf['prediction'])

result_dict = {'key': [], 
               'project': [],
               'dice_background': [],
               'dice_lesion': [],
               'volume_mask': [],
               'volume_prediction': []}
for key in keys:
    print(key)
    # load predictions
    with zarr.open(store=zarr.ZipStore(prediction_path, mode='r')) as zf:
        mask_predicted = zf[f'prediction/{key}'][0, :]

    # load training data
    with h5py.File(data_path, 'r') as hf:
        ds = hf[f'image/{key}']
        img = ds[0,:].astype(np.float32)
        project = ds.attrs['project']
        affine = ds.attrs['affine'] 
        mask = hf[f'mask_iso/{key}'][0, :]

    # store nifti files
    nib.save(nib.Nifti1Image(img, affine), 
                            (nifti_dir/f'{key}_img.nii.gz'))
    nib.save(nib.Nifti1Image(mask, affine),  
                            (nifti_dir/f'{key}_mask.nii.gz'))
    nib.save(nib.Nifti1Image(mask_predicted, affine),  
                            (nifti_dir/f'{key}_mask_predicted.nii.gz'))

    # evaluate dice
    oh_mask = expand_as_one_hot(torch.tensor(mask[np.newaxis, ...]).long(), C=2)
    oh_mask_predicted = expand_as_one_hot(torch.tensor(mask_predicted[np.newaxis, ...]).long(), C=2)
    dice = compute_per_channel_dice(oh_mask_predicted, oh_mask)

    # volumina
    voxel_vol = 2*2*3 # mm^3
    volume_mask = np.sum(mask)*voxel_vol
    volume_prediction = np.sum(mask_predicted)*voxel_vol

    # save dice results
    print(f'{key} dice {dice}')
    result_dict['key'].append(key)
    result_dict['project'].append(project)
    result_dict['volume_mask'].append(volume_mask)
    result_dict['volume_prediction'].append(volume_prediction)
    result_dict['dice_background'].append(dice[0])
    result_dict['dice_lesion'].append(dice[1])

    # plot validation figures
    fig, ax = plt.subplots(2,1, figsize=[10,15])
    ax[0].title.set_text('ground truth')
    ax[0].imshow(img.max(axis=1), cmap='gray', vmin=0, vmax=0.2)
    ax[0].axis('off')
    mip = mask.max(axis=1)
    mip = np.ma.array(mip, mask=(mip==0))
    ax[0].imshow(mip, alpha=0.8, vmin=0, cmap='coolwarm')

    ax[1].title.set_text(f'prediction \n DICE bg {dice[0]: .3f} lesion {dice[1]: .3f} project {project}')
    ax[1].axis('off')
    ax[1].imshow(img.max(axis=1), cmap='gray', vmin=0, vmax=0.2)
    mip = mask_predicted.max(axis=1)
    mip = np.ma.array(mip, mask=(mip==0))
    ax[1].imshow(mip, alpha=0.8, vmin=0, cmap='coolwarm')
    plt.tight_layout()
    plt.savefig(png_dir/f'{key}.png', bbox_inches='tight',pad_inches = 0)
    plt.close(fig)

# csv dice results to csv
df = pd.DataFrame.from_dict(result_dict)
df.to_csv(work_dir/'processed/petct/petct_validation.csv')
