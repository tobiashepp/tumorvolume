
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import zarr
import h5py
import matplotlib
import torch
import pandas as pd
import nibabel as nib
from pathlib import Path
import numpy as np
import nibabel as nib

from midasmednet.unet.loss import compute_per_channel_dice, expand_as_one_hot
from tumorvolume.ctorgans.postprocessing import largest_component

# Validation of the organ predictions for the test data 
# (with ground truth labels as reference).

# paths
work_dir = Path('/mnt/qdata/raheppt1/data/tumorvolume/')
data_path = work_dir/'interim/ctorgans/ctorgans.h5' 
prediction_path = work_dir/'processed/ctorgans/ctorgans_validation.zip'
png_dir = Path(work_dir/'processed/ctorgans/ctorgans_validation_png' )
nifti_dir =  work_dir/'processed/ctorgans/ctorgans_validation_nifti'
png_dir.mkdir(exist_ok=True)
nifti_dir.mkdir(exist_ok=True)

# get key list
with zarr.open(store=zarr.ZipStore(prediction_path, mode='r')) as zf:
    keys = list(zf['prediction'])

result_dict = {'key': [], 
               'dice_background': [],
               'dice_liver': [],
               'dice_spine': [],
               'dice_spleen': [],
               'dice_postprocessed_background': [],
               'dice_postprocessed_liver': [],
               'dice_postprocessed_spine': [],
               'dice_postprocessed_spleen': []}
for key in keys:
    # load predictions
    with zarr.open(store=zarr.ZipStore(prediction_path, mode='r')) as zf:
        mask_predicted = zf[f'prediction/{key}'][0, :]

    # load training data
    with h5py.File(data_path, 'r') as hf:
        ds = hf[f'images/{key}']
        img = ds[0,:].astype(np.float32)
        affine = ds.attrs['affine'] 
        mask = hf[f'labels/{key}'][0, :]

    # postprocess
    oh_mask = expand_as_one_hot(torch.tensor(mask[np.newaxis, ...]).long(), C=4)
    oh_mask_predicted = expand_as_one_hot(torch.tensor(mask_predicted[np.newaxis, ...]).long(), C=4)
    oh_mask_predicted_pp = largest_component(oh_mask_predicted.numpy()[0,...])
    mask_predicted_pp = np.argmax(oh_mask_predicted_pp, axis=0)
    oh_mask_predicted_pp = torch.tensor(oh_mask_predicted_pp[np.newaxis, ...])

    # store nifti files
    nib.save(nib.Nifti1Image(img, affine), 
                            (nifti_dir/f'{key}_img.nii.gz'))
    nib.save(nib.Nifti1Image(mask, affine),  
                            (nifti_dir/f'{key}_mask.nii.gz'))
    nib.save(nib.Nifti1Image(mask_predicted, affine),  
                            (nifti_dir/f'{key}_mask_predicted.nii.gz'))
    nib.save(nib.Nifti1Image(mask_predicted_pp, affine), 
                            (nifti_dir/f'{key}_mask_predicted_pp.nii.gz'))
   
    # evaluate dice
    dice = compute_per_channel_dice(oh_mask_predicted, oh_mask)
    dice_pp = compute_per_channel_dice(oh_mask_predicted_pp, oh_mask)

    # save dice results
    print(f'{key} dice {dice} dice/pp {dice_pp}')
    result_dict['key'].append(key)
    result_dict['dice_background'].append(dice[0])
    result_dict['dice_liver'].append(dice[1])
    result_dict['dice_spine'].append(dice[2])
    result_dict['dice_spleen'].append(dice[3])
    result_dict['dice_postprocessed_background'].append(dice_pp[0])
    result_dict['dice_postprocessed_liver'].append(dice_pp[1])
    result_dict['dice_postprocessed_spine'].append(dice_pp[2])
    result_dict['dice_postprocessed_spleen'].append(dice_pp[3])

    # plot validation figures
    fig, ax = plt.subplots(3,1, figsize=[10,15])
    ax[0].title.set_text('ground truth')
    ax[0].imshow(img.max(axis=1), cmap='gray')
    ax[0].axis('off')
    mip = mask.max(axis=1)
    mip = np.ma.array(mip, mask=(mip==0))
    ax[0].imshow(mip, alpha=0.5)

    ax[1].title.set_text(f'prediction \n DICE bg {dice[0]: .3f} liver {dice[1]: .3f} spine {dice[2]: .3f} spleen {dice[3]: .5f}')
    ax[1].axis('off')
    ax[1].imshow(img.max(axis=1), cmap='gray')
    mip = mask_predicted.max(axis=1)
    mip = np.ma.array(mip, mask=(mip==0))
    ax[1].imshow(mip, alpha=0.5)

    ax[2].title.set_text(f'prediction post processed\n DICE bg {dice_pp[0]: .3f} liver {dice_pp[1]: .3f} spine {dice_pp[2]: .3f} spleen {dice_pp[3]: .3f}')
    ax[2].axis('off')
    ax[2].imshow(img.max(axis=1), cmap='gray')
    mip = mask_predicted_pp.max(axis=1)
    mip = np.ma.array(mip, mask=(mip==0))
    ax[2].imshow(mip, alpha=0.5)
    plt.tight_layout()
    plt.savefig(png_dir/f'{key}.png', bbox_inches='tight',pad_inches = 0, facecolor='black')
    plt.close(fig)

# csv dice results to csv
df = pd.DataFrame.from_dict(result_dict)
df.to_csv(work_dir/'processed/ctorgans/ctorgans_validation.csv')
df.to_feather(work_dir/'processed/ctorgans/ctorgans_validation.feather')
