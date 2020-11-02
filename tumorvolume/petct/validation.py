import os
from pathlib import Path

import hydra
import zarr
import h5py
import matplotlib
matplotlib.use('Agg')
import torch
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import numpy as np
import nibabel as nib

from p_tqdm import p_map
from scipy.ndimage import label

from midasmednet.unet.loss import compute_per_channel_dice, expand_as_one_hot
from tumorvolume.ctorgans.postprocessing import largest_component

@hydra.main(config_path=os.getenv('CONFIG'), strict=False)
def validation(cfg):
    """Validate lesion segmentation (use manual labels as reference).

    Args:
        cfg (OmegaConf): Hydra configuration.
    """
    # copy over
    test_set = cfg.prediction.test_set
    image_group = cfg.base.image_group
    label_group = cfg.base.label_group
    voxel_vol = cfg.validation.voxel_vol
    petsuv_scale = cfg.validation.petsuv_scale
    data_path = Path(cfg.base.data)
    prediction_path = Path(cfg.validation.data) 
    prediction_group = cfg.validation.group
    png_dir = Path(cfg.validation.plots)/f'{cfg.base.name}_predictions{cfg.base.suffix}'
    nifti_dir = Path(cfg.validation.export)/'nifti'/f'{cfg.base.name}_predictions{cfg.base.suffix}'
    png_dir.mkdir(exist_ok=True, parents=True)
    nifti_dir.mkdir(exist_ok=True, parents=True)
    csv_path = Path(cfg.validation.export)/f'{cfg.base.name}_predictions{cfg.base.suffix}.csv'
    
    if prediction_path.suffix == '.zip':
        store = zarr.ZipStore(prediction_path, mode='r')
    else:
        store = zarr.DirectoryStore(prediction_path)

    # get key list
    with zarr.open(store=store) as zf:
        keys = list(zf[f'{prediction_group}'])

    result_dict = {'key': [], 
                'project': [],
                'dice_background': [],
                'dice_lesion': [],
                'volume_manual': [],
                'volume_prediction': []}

    def proc(key):
        # load predictions
        
        with zarr.open(store=store) as zf:
            mask_predicted = zf[f'{prediction_group}/{key}'][0, :]

        # load training data
        with h5py.File(data_path, 'r') as hf:
            ds = hf[f'{image_group}/{key}']
            img = ds[0,:].astype(np.float32)
            project = ds.attrs['project']
            affine = ds.attrs['affine'] 
            mask = hf[f'{label_group}/{key}'][0, :]

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
        volume_manual = np.sum(mask)*voxel_vol
        volume_prediction = np.sum(mask_predicted)*voxel_vol

        # calculate uptake and number of lesions
        uptake_manual = np.ma.array(mask*img, mask=mask==0)*petsuv_scale
        uptake_prediction = np.ma.array(mask_predicted*img, mask=mask_predicted==0)*petsuv_scale
        _, num_lesions_manual= label(mask)
        _, num_lesions_prediction = label(mask_predicted)

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
        
        result = [key, project, 
                  volume_manual, volume_prediction, 
                  dice[0].item(), dice[1].item(),
                  uptake_manual.mean(),
                  uptake_manual.std(),
                  np.ma.median(uptake_manual.mean()),
                  uptake_manual.min(),
                  uptake_manual.max(),
                  num_lesions_manual,
                  uptake_prediction.mean(),
                  uptake_prediction.std(),
                  np.ma.median(uptake_prediction.mean()),
                  uptake_prediction.min(),
                  uptake_prediction.max(),
                  num_lesions_prediction]

        return result

    # multiprocessing
    results = p_map(proc, keys, num_cpus=cfg.validation.jobs)
    results = list(zip(*results))

    # save results
    result_dict['key'] = results[0]
    result_dict['project'] = results[1]
    result_dict['volume_manual'] = results[2]
    result_dict['volume_prediction'] = results[3]
    result_dict['dice_background'] = results[4]
    result_dict['dice_lesion'] = results[5] 
    result_dict['uptake_mean_manual'] = results[6] 
    result_dict['uptake_std_manual'] = results[7] 
    result_dict['uptake_median_manual'] = results[8] 
    result_dict['uptake_min_manual'] = results[9] 
    result_dict['uptake_max_manual'] = results[10] 
    result_dict['num_lesions_manual'] = results[11] 
    result_dict['uptake_mean_prediction'] = results[12] 
    result_dict['uptake_std_prediction'] = results[13] 
    result_dict['uptake_median_prediction'] = results[14] 
    result_dict['uptake_min_prediction'] = results[15] 
    result_dict['uptake_max_prediction'] = results[16] 
    result_dict['num_lesions_prediction'] = results[17] 

    # csv dice results to csv
    df = pd.DataFrame.from_dict(result_dict)
    df.to_csv(csv_path)


if __name__ == '__main__':
    validation()