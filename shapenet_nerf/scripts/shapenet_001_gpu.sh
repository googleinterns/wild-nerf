#!/bin/bash

/google/bin/releases/xmanager/cli/xmanager.par launch \
  experimental/users/daeyun/jaxnerf/xm_launcher.py -- \
  --xm_deployment_env=alphabet \
  --xm_resource_alloc=group:peace/peace-interns \
  --xm_resource_pool=peace \
  --xm_skip_launch_confirmation \
  --cell=lu \
  --cell_eval=lu \
  --n_gpus=4 \
  --nouse_tpu \
  --data_dir="/cns/lu-d/home/daeyun/shapenet/v2/train/00002/" \
  --train_dir="/cns/lu-d/home/daeyun/checkpoints/shapenet/v2/0002/" \
  --priority=200 \
  --config=jaxnerf/configs/shapenet_base \
  --tags="shapenet,v2" \
  "$@"
