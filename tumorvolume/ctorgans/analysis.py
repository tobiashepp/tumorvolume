import os
import matplotlib
import zipfile
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import basename
from zipfile import ZipFile

import click
import torch
import zarr
import h5py
import numpy as np
import pandas as pd
from p_tqdm import p_map
from pathlib import Path
from skimage.measure import label 
from scipy.ndimage import find_objects
from dotenv import load_dotenv

from tumorvolume.ctorgans.postprocessing import one_hot_encoded

load_dotenv()
DATA = os.getenv('DATA')

@click.command()
@click.option('--jobs', default=1, help='number of parallel jobs')
@click.option('--prediction_path', default='DATA/processed/ctorgans/ctorgans_petct_1.zarr')
@click.option('--data_path', default='DATA/interim/petct/TUE0000ALLDS_3D.h5')
def analysis(prediction_path, data_path, jobs):
    # paths
    work_dir = Path(DATA)
    data_path = Path(data_path.replace('DATA', DATA))
    prediction_path = Path(prediction_path.replace('DATA', DATA))
    out_dir = work_dir/'processed/ctorgans'/f'{prediction_path.stem}_png'
    out_dir.mkdir(exist_ok=True)

    # get key list
    with zarr.open(str(prediction_path)) as zf:
        keys = list(zf['prediction'])

    print(f'analyzing {len(keys)} subjects in {prediction_path}')     
    def proc(key):
        results = {}

        # load predictions
        with zarr.open(store=zarr.DirectoryStore(prediction_path)) as zf:
            mask_predicted = zf[f'processed/{key}'][0, :]

        # load training data
        with h5py.File(data_path, 'r') as hf:
            ds = hf[f'image/{key}']
            img = ds[1,:].astype(np.float32)
            pet = ds[0,:].astype(np.float32)
            project = ds.attrs['project']

        # postprocess
        one_hot_mask = one_hot_encoded(mask_predicted, C=4)

        # analysis 
        voxel_vol = 2*2*3 # mm^3
        petsuv_scale = 40
        mask_liver = one_hot_mask[1]
        mask_spine = one_hot_mask[2]
        mask_spleen = one_hot_mask[3]
        results['key'] = [str(key)]
        results['project'] = [str(project)]
        results['volume_liver'] = [float(np.sum(mask_liver)*voxel_vol)]
        results['volume_spine'] = [float(np.sum(mask_spine)*voxel_vol)]
        results['volume_spleen'] = [float(np.sum(mask_spleen)*voxel_vol)]

        uptake_liver = np.ma.array(mask_liver*pet, mask=(mask_liver==0))
        results['uptake_liver_mean'] = [float(uptake_liver.mean()*petsuv_scale)]
        results['uptake_liver_std'] = [float(uptake_liver.std()*petsuv_scale)]
        results['uptake_liver_min'] = [float(uptake_liver.min()*petsuv_scale)]
        results['uptake_liver_max'] = [float(uptake_liver.max()*petsuv_scale)]
        results['uptake_liver_median'] = [float(np.ma.median(uptake_liver)*petsuv_scale)]

        uptake_spine = np.ma.array(mask_spine*pet, mask=(mask_spine==0))
        results['uptake_spine_mean'] = [float(uptake_spine.mean()*petsuv_scale)]
        results['uptake_spine_std'] = [float(uptake_spine.std()*petsuv_scale)]
        results['uptake_spine_min'] = [float(uptake_spine.min()*petsuv_scale)]
        results['uptake_spine_max'] = [float(uptake_spine.max()*petsuv_scale)]
        results['uptake_spine_median'] = [float(np.ma.median(uptake_spine)*petsuv_scale)]

        uptake_spleen = np.ma.array(mask_spleen*pet, mask=(mask_spleen==0))
        results['uptake_spleen_mean'] = [float(uptake_spleen.mean()*petsuv_scale)]
        results['uptake_spleen_std'] = [float(uptake_spleen.std()*petsuv_scale)]
        results['uptake_spleen_min'] = [float(uptake_spleen.min()*petsuv_scale)]
        results['uptake_spleen_max'] = [float(uptake_spleen.max()*petsuv_scale)]
        results['uptake_spleen_median'] = [float(np.ma.median(uptake_spleen)*petsuv_scale)]

        # plots
        obj = find_objects(mask_predicted==3)
        if len(obj) > 0:
            spleen_box = obj[0]
            spleen_pos = (spleen_box[2].start + spleen_box[2].stop)//2
        else:
            spleen_pos = 100
            
        fig, ax = plt.subplots(figsize=[10,15])
        ax.title.set_text(f"{key}\nSUV liver:{results['uptake_liver_mean'][0]: .3f} "\
                                + f"spine:{results['uptake_spine_mean'][0]: .3f} "\
                                + f"spleen:{results['uptake_spleen_mean'][0]: .3f} ")
        ax.imshow(pet.max(axis=1), cmap='gray', vmin=0.0, vmax=0.4)
        ax.axis('off')
        mip = (pet*mask_predicted).max(axis=1)
        mip = np.ma.array(mip, mask=(mip==0))
        ax.imshow(mip, alpha=0.5, cmap='inferno', vmin=0.0, vmax=0.2 )
        plt.tight_layout()
        plt.savefig(out_dir/f'{key}_pet.png', bbox_inches='tight',
                    pad_inches = 0, facecolor='black')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=[10,15])
        ax.title.set_text(f'{key}')
        ax.imshow(img[:,:,spleen_pos], cmap='gray', vmin=0.0, vmax=1.0)
        ax.axis('off')
        mip = mask_predicted[:,:,spleen_pos]
        mip = np.ma.array(mip, mask=(mip==0))
        ax.imshow(mip, alpha=0.5, cmap='inferno',vmin=0, vmax=3)
        plt.tight_layout()
        plt.savefig(out_dir/f'{key}_slice.png', bbox_inches='tight',
                    pad_inches = 0, facecolor='black')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=[10,15])
        ax.title.set_text(f'{key}')
        ax.imshow(img.max(axis=1), cmap='gray', vmin=0.0, vmax=1.0)
        ax.axis('off')
        mip = (mask_predicted).max(axis=1)
        mip = np.ma.array(mip, mask=(mip==0))
        ax.imshow(mip, alpha=0.5, cmap='inferno',vmin=0, vmax=3)
        plt.tight_layout()
        plt.savefig(out_dir/f'{key}_ct.png', bbox_inches='tight',
                    pad_inches = 0, facecolor='black')
        plt.close(fig)

        return results

    # parallel processing
    results = p_map(proc, keys, num_cpus=jobs)

    # save dataframe with results
    df = pd.concat([pd.DataFrame.from_dict(r) for r in results])
    df = df.reset_index()
    df.to_feather(work_dir/'processed/ctorgans'/f'{prediction_path.stem}.bin')

    # zip results 
    print('zipping results ...')
    zip_path = work_dir/'processed/ctorgans'/f'{prediction_path.stem}_png.zip'
    with ZipFile(zip_path, 'w', compression=zipfile.ZIP_LZMA) as zipf:
        for folder, subfolders, filenames in os.walk(out_dir):
            for filename in filenames:
                p = os.path.join(folder, filename)
                # add file to zip
                zipf.write(p, basename(p))

if __name__ == '__main__':
    analysis()