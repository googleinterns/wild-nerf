#!/bin/bash

set -ex

GPU="$1"
COLAB_NAME="$2"

# https://g3doc.corp.google.com/learning/brain/frameworks/g3doc/tools/colab/README.md?cl=head#custom
/google/bin/releases/xmanager/cli/xmanager.par launchborg \
  learning/deepmind/public/tools/ml_python/ml_notebook.borg -- \
    --vars="notebook_binary=$HOME/nerf_runtime.par,cell=lu,registration_label='${COLAB_NAME}',gpu_tesla_v100=${GPU}" \
    --xm_deployment_env=alphabet \
    --xm_resource_alloc=group:peace/peace-interns \
    --xm_resource_pool=peace \
    --xm_skip_launch_confirmation \
    --experiment_name="${COLAB_NAME}" \
    --experiment_tags='idle_opt_out=colab'
