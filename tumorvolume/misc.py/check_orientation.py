import h5py
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.measure import label
from scipy.ndimage import find_objects

h5_path = '/mnt/qdata/raheppt1/data/tumorvolume/interim/ctorgans/ctorgans.h5'
png_dir = Path('/mnt/qdata/raheppt1/data/tumorvolume/interim/ctorgans/validation')
with h5py.File(h5_path, 'r') as hf:
    target_keys = list(hf['images'])
    for key in target_keys:
        print(key)
        img = hf[f'images/{key}'][0, :].astype(np.float32)
        mask = hf[f'labels/{key}'][0, :].astype(np.uint8)
        spleen_box = find_objects(mask==3)[0]
        fig, ax = plt.subplots(figsize=[10,15])
        ax.title.set_text('ground truth')
        axis=1
        ax.imshow(img.max(axis=axis), cmap='gray')
        mip = mask.max(axis=axis)
        mip = np.ma.array(mip, mask=(mip==0))
        ax.imshow(mip, alpha=0.5)
        plt.savefig(png_dir/f'frontal{key}.png')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=[10,15])
        ax.title.set_text('ground truth')
        axis=2
        ax.imshow(img.max(axis=axis), cmap='gray')
        mip = mask.max(axis=axis)
        mip = np.ma.array(mip, mask=(mip==0))
        ax.imshow(mip, alpha=0.5)
        plt.savefig(png_dir/f'axial{key}.png')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=[10,15])
        ax.title.set_text('ground truth')
        axis=0
        ax.imshow(img.max(axis=axis), cmap='gray')
        mip = mask.max(axis=axis)
        mip = np.ma.array(mip, mask=(mip==0))
        ax.imshow(mip, alpha=0.5)
        plt.savefig(png_dir/f'sagittal{key}.png')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=[10,15])
        ax.title.set_text('ground truth')
        mip = mask[:,:,(spleen_box[2].start +spleen_box[2].stop)//2]
        mip = np.ma.array(mip, mask=(mip==0))
        ax.imshow(img[:,:,(spleen_box[2].start +spleen_box[2].stop)//2], cmap='gray')
        ax.imshow(mip, alpha=0.3)
        plt.savefig(png_dir/f'slice{key}.png')
        plt.close(fig)
        #break