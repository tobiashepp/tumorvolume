import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
DATA = os.getenv('DATA')

path_h5 = Path(DATA)/'interim/petct/TUE0000ALLDS_3D.h5'
path_h5_out = Path(DATA)/'interim/petct/TUE0000ALLDS_3D_fixed.h5'

# copy data from path_h5 to path_h5_out, all arrays are zero padded
with h5py.File(path_h5, mode='r') as hf:
    with h5py.File(path_h5_out, mode='w') as hf_out:
        keys = list(hf['image'])
        for key in tqdm(keys):
            for cat in list(hf):
                gr = hf_out.require_group(cat)
                ds_in = hf[cat][key]
                data = ds_in[:]
                if cat in ['mask', 'mask_iso']:
                    data = data[np.newaxis, ...]
                ds_out = gr.require_dataset(key, data.shape, data.dtype)
                ds_out[:] = data
                for (k, val) in ds_in.attrs.items():
                    ds_out.attrs[k] = val