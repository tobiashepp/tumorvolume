import os
from pathlib import Path

import h5py
from sklearn.model_selection import KFold
from dotenv import load_dotenv
load_dotenv()

# create cross validation key files 
DATA = os.getenv('DATA')
h5_path = Path(DATA)/'interim/petbc/TUE0000ALLDS_3D.h5'
shuffle=True
splits = 5
with h5py.File(h5_path, 'r') as hf:
    subject_keys = list(hf['image'])
kf = KFold(n_splits=5, shuffle=shuffle)
for k, (train_index, test_index) in enumerate(kf.split(subject_keys)):
    train_keys = [(subject_keys[i] + '\n') for i in train_index]
    with Path(h5_path.parent/(h5_path.stem+f'_train{k}.dat')).open('w') as keyfile:
        keyfile.writelines(train_keys)
    test_keys = [(subject_keys[i] + '\n') for i in test_index]
    with Path(h5_path.parent/(h5_path.stem+f'_test{k}.dat')).open('w') as keyfile:
        keyfile.writelines(test_keys)
