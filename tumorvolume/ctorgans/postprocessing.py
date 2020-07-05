import os
import sys
from pathlib import Path

import zarr
import click
import numpy as np
from dotenv import load_dotenv
from p_tqdm import p_map
from skimage.measure import label 
from scipy.ndimage.morphology import binary_fill_holes
load_dotenv()

def largest_component(one_hot_mask):
    """Select the largest connected component for each channel of the mask.

    Args:
        one_hot_mask (np.array): mask label (one hot encoded), CxHxWxD

    Returns:
        np.array: Processed mask
    """
    # mask as one hot encoded CxHxWxD
    # select largest component
    pp_mask = []
    pp_mask.append(one_hot_mask[0])
    for channel in range(1, len(one_hot_mask)):
        mask = one_hot_mask[channel, ...]
        mask_l, num_of_comp  = label(mask, return_num=True)
        if num_of_comp > 0:
            comp_size = [(mask_l==c).sum() for c in range(num_of_comp + 1)]
            largest_comp = np.argmax(comp_size[1:]) + 1
            mask_l = (mask_l==largest_comp).astype(np.uint8)
            pp_mask.append(mask_l)
        else:
            pp_mask.append(mask)
    pp_mask = np.stack(pp_mask, axis=0)
    return pp_mask


def fill_holes(one_hot_mask):
    """ Binary fill holes for each channel for the mask.

    Args:
        one_hot_mask (np.array): mask label (one hot encoded), CxHxWxD

     Returns:
        np.array: Processed mask
    """
    pp_mask = []
    pp_mask.append(one_hot_mask[0])
    for channel in range(1, len(one_hot_mask)):
        m = binary_fill_holes(one_hot_mask[channel]).astype(np.uint8) 
        pp_mask.append(m)
    return np.stack(pp_mask, axis=0)


def one_hot_encoded(categorial_mask, C):
    """Convert categorial encoding to binary one hot encoding.
    
    Args:
        categorial_mask (np.array): mask label HxWxD
        C (int): number of classes

     Returns:
        np.array:  one_hot_mask (CxHxWxD)
    """
    return np.eye(C)[categorial_mask].transpose([3,0,1,2])


@click.command()
@click.option('--jobs', default=1, help='number of parallel jobs')
@click.argument('prediction_path')
def run_postprocessing(prediction_path, jobs): 
    DATA = os.getenv('DATA')
    prediction_path = str(prediction_path).replace('DATA', DATA)
    prediction_path = Path(prediction_path)
    print(f'post processing {prediction_path}')

    with zarr.open(store=zarr.DirectoryStore(prediction_path)) as zf:
        # read from group 'prediction', store to group 'processed'
        gr = zf['prediction']
        gr_pp = zf.require_group('processed')
        keys = list(gr)
        
        # postprocessing: largest componente/fill holes
        def proc(key):   
            ds = gr[key]
            mask = ds[0, :]
            affine = ds.attrs['affine']
            one_hot_mask = one_hot_encoded(mask, C=4)
            one_hot_mask = fill_holes(one_hot_mask)
            one_hot_mask = largest_component(one_hot_mask)
            mask_pp = np.argmax(one_hot_mask, axis=0)
            mask_pp = mask_pp[np.newaxis, ...]
            ds_pp = gr_pp.require_dataset(key, mask_pp.shape, dtype=mask_pp.dtype)
            ds_pp[:] = mask_pp
            ds_pp.attrs['affine'] = affine

        # parallel processing
        p_map(proc, keys, num_cpus=jobs)
    

if __name__ == '__main__':
    run_postprocessing()