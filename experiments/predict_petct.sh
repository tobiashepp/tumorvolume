#!/bin/bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
echo "$(pyenv which python)"
export PRJ=/home/raheppt1/projects
export DATA=/mnt/qdata/raheppt1/data/tumorvolume

pyenv deactivate
pyenv activate mednet
export CONFIG=$PRJ/tumorvolume/config/petct.yaml
export CUDA_VISIBLE_DEVICES=0
python $PRJ/mednet/examples/predict.py prediction.test_set="$DATA/interim/petct/keys/TUE0000ALLDS_3D_test0.dat" prediction.checkpoint="/mnt/share/raheppt1/data/tumorvolume/petct/models/default_petct/17_MED-169/checkpoints/epoch=145.ckpt"
python $PRJ/mednet/examples/predict.py prediction.test_set="$DATA/interim/petct/keys/TUE0000ALLDS_3D_test1.dat" prediction.checkpoint="/mnt/share/raheppt1/data/tumorvolume/petct/models/default_petct/18_MED-174/checkpoints/epoch=119.ckpt"
python $PRJ/mednet/examples/predict.py prediction.test_set="$DATA/interim/petct/keys/TUE0000ALLDS_3D_test2.dat" prediction.checkpoint="/mnt/share/raheppt1/data/tumorvolume/petct/models/default_petct/19_MED-175/checkpoints/epoch=111.ckpt"
python $PRJ/mednet/examples/predict.py prediction.test_set="$DATA/interim/petct/keys/TUE0000ALLDS_3D_test3.dat" prediction.checkpoint="/mnt/share/raheppt1/data/tumorvolume/petct/models/default_petct/20_MED-176/checkpoints/epoch=119.ckpt"
python $PRJ/mednet/examples/predict.py prediction.test_set="$DATA/interim/petct/keys/TUE0000ALLDS_3D_test4.dat" prediction.checkpoint="/mnt/share/raheppt1/data/tumorvolume/petct/models/default_petct/21_MED-177/checkpoints/epoch=113.ckpt"
