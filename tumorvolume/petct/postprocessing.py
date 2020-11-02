import os
from pathlib import Path

import hydra
import h5py
import zarr
import dotenv
import scipy.ndimage.morphology
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from p_tqdm import p_map
from skimage.measure import label

dotenv.load_dotenv()

def lcomp(mask):
    """Computes largest connected component for binary mask.
    
    Args:
        mask (np.array): input binary mask
    
    Returns:
        np.array: largest connected component
    """

    labels = label(mask)
    unique, counts = np.unique(labels, return_counts=True)
    # the 0 label is by default background so take the rest
    list_seg = list(zip(unique, counts))[1:]
    largest = max(list_seg, key=lambda x: x[1])[0]
    labels_max = (labels == largest).astype(np.uint8)
    return labels_max

@hydra.main(config_path=os.getenv('CONFIG'), strict=False)
def main(cfg):
    image_data = cfg.base.data 
    image_group = cfg.base.image_group
    prediction_data = cfg.prediction.data
    prediction_group = cfg.prediction.group
    postprocessed_data = cfg.postprocessing.data 
    postprocessed_group = cfg.postprocessing.group
    threshold = cfg.postprocessing.threshold

    # get keys
    with zarr.open(store=zarr.ZipStore(prediction_data), mode='r') as zf:
        keys = list(zf[prediction_group])


    with zarr.open(store=zarr.DirectoryStore(postprocessed_data), mode='w') as outf:
        def proc(key):
            with h5py.File(image_data, 'r') as hf:
                img = hf[image_group][key][:]
            with zarr.open(store=zarr.ZipStore(prediction_data), mode='r') as zf:
                ds = zf[prediction_group][key]
                prediction = ds[:]
                affine = ds.attrs['affine']

            # threshold ct
            img_th = img[1]>threshold
            # fill holes per slice
            for k in range(img_th.shape[2]):
                img_th[:,:,k] = scipy.ndimage.morphology.binary_fill_holes(img_th[:,:,k])
            # keep largest connected component
            img_th = lcomp(img_th)

            prediction[0, ...] = (prediction[0, ...]*img_th).astype(np.uint8)
            gr = outf.require_group(postprocessed_group)
            ds_out = gr.require_dataset(key, shape=prediction.shape, dtype=prediction.dtype, chunks=False)
            ds_out[:] = prediction
        
        p_map(proc, keys, num_cpus=cfg.postprocessing.jobs)

if __name__ == '__main__':
    main()