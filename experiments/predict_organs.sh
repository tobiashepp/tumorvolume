#!/bin/bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
echo "$(pyenv which python)"
export PRJ=/home/raheppt1/projects/tumorvolume
export DATA=/mnt/qdata/raheppt1/data/tumorvolume
export CONFIG=$PRJ/config/ctorgans_predict_petct.yaml
export VENV=tumorvolume
export CUDA_VISIBLE_DEVICES=0

conda activate $VENV
python $PRJ/torch-mednet/examples/predict.py
