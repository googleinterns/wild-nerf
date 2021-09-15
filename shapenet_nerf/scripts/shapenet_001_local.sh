#!/bin/bash

blaze run \
  --compilation_mode=opt \
  --define=PYTYPE=TRUE \
  --symlink_prefix=/ -- \
experimental/users/daeyun/jaxnerf/train \
  --logtostderr \
  --data_dir="/cns/lu-d/home/daeyun/shapenet/v2/train/00002/" \
  --train_dir="/cns/lu-d/home/daeyun/checkpoints/shapenet/v2/0002/" \
  --config=jaxnerf/configs/shapenet_base \
  "$@"
