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
  --config="jaxnerf/configs/llff_360" \
  --config_extra="objectron_barf/base,objectron_barf/extrinsics_007" \
  --xm_tags="objectron,batch-24/33,plain,lowerlr" \
  --data_dir="/cns/lu-d/home/daeyun/objectron/video_frames/images_8/chair/batch-24/33" \
  --train_dir="/cns/lu-d/home/daeyun/checkpoints/batch-24/33/0013/" \
  "$@"
