import os
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

@hydra.main(config_path='/home/raheppt1/projects/tumorvolume/config/petct.yaml', strict=False)
def nora(cfg):
    """Export results to nora project.

    Args:
        cfg (OmegaConf): Hydra configuration.
    """
    # copy over
    image_group = cfg.base.image_group
    label_group = cfg.base.label_group
    data_path = Path(cfg.base.data)
    prediction_path = Path(cfg.prediction.data) 

    # nora parameters
    nora_path = Path(os.getenv('NORA'))
    nora_project = cfg.nora_project or 'raheppt1___petct'
    nora_study = cfg.nora_study or '0001'
    nora_date = cfg.nora_date or '20200701'

    # get key list
    with zarr.open(store=zarr.ZipStore(prediction_path, mode='r')) as zf:
        keys = list(zf['prediction'])

    for key in tqdm(keys):

        # create study directory
        study_dir = nora_path/nora_project/f'noname_noname_{key}'/f'study{nora_study}_{nora_date}'
        study_dir.mkdir(exist_ok=True, parents=True)

        # load predictions
        with zarr.open(store=zarr.ZipStore(prediction_path, mode='r')) as zf:
            mask_predicted = zf[f'prediction/{key}'][0, :]

        # load training data
        with h5py.File(data_path, 'r') as hf:
            ds = hf[f'{image_group}/{key}']
            img = ds[0,:].astype(np.float32)
            project = ds.attrs['project']
            affine = ds.attrs['affine'] 
            mask = hf[f'{label_group}/{key}'][0, :]

        # store nifti files
        nib.save(nib.Nifti1Image(img, affine), 
                                (study_dir/f'{key}_img.nii.gz'))
        nib.save(nib.Nifti1Image(mask, affine),  
                                (study_dir/f'{key}_mask.nii.gz'))
        nib.save(nib.Nifti1Image(mask_predicted, affine),  
                                (study_dir/f'{key}_mask_predicted.nii.gz'))

        # add to nora
        subprocess.run(['nora','-p',nora_project,'--add',study_dir])

if __name__ == '__main__':
    nora()