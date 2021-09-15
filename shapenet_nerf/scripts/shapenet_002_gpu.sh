#!/bin/bash

set -ex

/google/bin/releases/xmanager/cli/xmanager.par launch \
  experimental/users/daeyun/jaxnerf/xm_launcher.py -- \
  --xm_deployment_env=alphabet \
  --xm_resource_alloc=group:peace/peace-interns \
  --xm_resource_pool=peace \
  --xm_skip_launch_confirmation \
  --cell=lu \
  --cell_eval=lu \
  --n_gpus=8 \
  --nouse_tpu \
  --priority=200 \
  --config="jaxnerf/configs/shapenet_base" \
  --config_extra="shapenet_barf/base,shapenet_barf/extrinsics_004" \
  --xm_tags="shapenet,v5,train/00002" \
  --data_dir="/cns/lu-d/home/daeyun/shapenet/v5/train/00002.pkl" \
  --train_dir="/cns/lu-d/home/daeyun/checkpoints/shapenet/v5/train/00002/004/" \
  "$@"
