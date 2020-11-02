#!/bin/bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
echo "$(pyenv which python)"
export PRJ=/home/raheppt1/projects
export DATA=/mnt/qdata/raheppt1/data/tumorvolume

pyenv deactivate
pyenv activate mednet
export CONFIG=$PRJ/tumorvolume/config/ctorgans_predict_petct.yaml
export CUDA_VISIBLE_DEVICES=0
python $PRJ/mednet/examples/predict.py
