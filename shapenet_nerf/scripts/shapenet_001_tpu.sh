#!/bin/bash

/google/bin/releases/xmanager/cli/xmanager.par launch \
  experimental/users/daeyun/jaxnerf/xm_launcher.py -- \
  --xm_deployment_env=alphabet \
  --xm_resource_alloc=group:peace/peace-interns \
  --xm_resource_pool=peace \
  --xm_skip_launch_confirmation \
  --cell=lu \
  --cell_eval=lu \
  --data_dir="/cns/lu-d/home/daeyun/shapenet/v2/train/00002/" \
  --train_dir="/cns/lu-d/home/daeyun/checkpoints/shapenet/v2/0002/" \
  --use_tpu \
  --tpu_topology=4x4 \
  --tpu_platform=jf \
  --use_tpu_eval \
  --tpu_topology_eval=4x4 \
  --tpu_platform_eval=jf \
  --priority=115 \
  --config=jaxnerf/configs/shapenet_base \
  "$@"
