#!/bin/bash

/google/bin/releases/xmanager/cli/xmanager.par launch \
  third_party/google_research/google_research/jaxnerf/google/xm/xm_launcher.py -- \
  --xm_deployment_env=alphabet \
  --xm_resource_alloc=group:peace/peace-interns \
  --xm_resource_pool=peace \
  --xm_skip_launch_confirmation \
  --cell=lu \
  --cell_eval=lu \
  --n_gpus=8 \
  --nouse_tpu \
  --data_dir="/cns/lu-d/home/daeyun/objectron/video_frames/images_8/chair/batch-24/33" \
  --train_dir="/cns/lu-d/home/daeyun/checkpoints/0001" \
  --priority=115 \
  --config=google/configs/llff_360
