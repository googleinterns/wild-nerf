#!/bin/bash

set -ex

blaze build -c opt --copt=-mavx --config=cuda --define cuda_target_sm75=1 \
    --verbose_failures \
    //experimental/users/daeyun/colab:nerf_runtime.par

cp ./blaze-bin/experimental/users/daeyun/colab/nerf_runtime.par ~/
