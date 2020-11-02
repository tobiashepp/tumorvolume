import os
import shutil
import subprocess
from pathlib import Path

import hydra
import zarr
import h5py
import numpy as np
import nibabel as nib
from tqdm import tqdm
from dotenv import load_dotenv

from midasmednet.unet.loss import compute_per_channel_dice, expand_as_one_hot
from tumorvolume.ctorgans.postprocessing import largest_component

# load .env configuration
load_dotenv()

@hydra.main(config_path=os.getenv('CONFIG'), strict=False)
def nora(cfg):
    """Export results to nora project.

    Args:
        cfg (OmegaConf): Hydra configuration.
    """
    # copy over
    test_set = cfg.prediction.test_set
    image_group = cfg.base.image_group
    label_group = cfg.base.label_group
    data_path = Path(cfg.base.data)
    prediction_path = Path(cfg.postprocessing.data) 
    prediction_group = cfg.postprocessing.group
    png_dir = Path(cfg.validation.plots)/f'{cfg.base.name}_predictions{cfg.base.suffix}'

    # nora parameters
    nora_path = Path(os.getenv('NORA'))
    nora_project = cfg.nora.nora_project or 'raheppt1___petct'
    nora_study = cfg.nora.nora_study or '0001'
    nora_date = cfg.nora.nora_date or '20200701'
    copy_png  = cfg.nora.nora_png or False
    copy_organs = cfg.nora.nora_organs or None

    # get key list
    with open(test_set, 'r') as f:
        keys = [l.strip() for l in f.readlines()]

    for key in tqdm(keys):
        # create study directory
        study_dir = nora_path/nora_project/f'noname_noname_{key}'/f'study{nora_study}_{nora_date}'
        study_dir.mkdir(exist_ok=True, parents=True)

        # load predictions
        with zarr.open(store=zarr.DirectoryStore(prediction_path)) as zf:
            mask_predicted = zf[f'{prediction_group}/{key}'][0, :]

        # load training data
        with h5py.File(data_path, 'r') as hf:
            ds = hf[f'{image_group}/{key}']
            img_pet = ds[0,:].astype(np.float32)
            img_ct = ds[1,:].astype(np.float32)
            project = ds.attrs['project']
            affine = ds.attrs['affine'] 
            mask = hf[f'{label_group}/{key}'][0, :]
        
        # store nifti files
        nib.save(nib.Nifti1Image(img_pet, affine), 
                                (study_dir/f'{key}_img_pet.nii.gz'))
        nib.save(nib.Nifti1Image(img_ct, affine), 
                                (study_dir/f'{key}_img_ct.nii.gz'))
        nib.save(nib.Nifti1Image(mask, affine),  
                                (study_dir/f'{key}_mask.nii.gz'))
        nib.save(nib.Nifti1Image(mask_predicted, affine),  
                                (study_dir/f'{key}_mask_predicted.nii.gz'))

        # copy validation png to study dir
        if copy_png:
            shutil.copy(png_dir/f'{key}.png', study_dir)

        # load ct organ predictions? 
        if copy_organs:
            with zarr.open(copy_organs) as zf:
                mask_organs = zf['processed'][key][0, :].astype(np.float32)
                affine = zf['processed'][key].attrs['affine']
                for c in range(1,4):
                    nib.save(nib.Nifti1Image((mask_organs == c).astype(np.uint8), affine),  
                                (study_dir/f'{key}_organ{c}_mask_predicted.nii.gz'))
        
        # add to nora
        subprocess.run(['nora','-p',nora_project,'--add',study_dir])

if __name__ == '__main__':
    nora()