load("//research/colab:build_defs.bzl", "colab_binary")

colab_binary(
    name = "nerf_runtime",
    extends = ["//learning/deepmind/dm_python:dm_notebook3"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//devtools/bauhaus/api:client_py_pb2",
        "//experimental/users/daeyun/jaxnerf/nerf:utils_extra",
        "//experimental/users/daeyun/objectron_nerf:camera_utils",
        "//experimental/users/daeyun/objectron_nerf:io_utils",
        "//experimental/users/daeyun/objectron_nerf:objectron_utils",
        "//experimental/users/daeyun/objectron_nerf:video_utils",
        "//file/colossus/public:cns",
        "//file/google_src",
        "//file/liball",
        "//file/localfile",
        "//file/placer",
        "//learning/brain/python/client:notebook_deps",
        "//learning/brain/research/jax:gpu_support",
        "//learning/brain/research/jax:tpu_support",
        "//learning/brain/research/jaxnerf/nerf",
        "//learning/deepmind/analysis/tensortree",
        "//learning/deepmind/analysis/tensortree:hlog",
        "//learning/deepmind/dm_python:deep_reloader",
        "//learning/deepmind/python/adhoc_import:binary_import",
        "//learning/deepmind/python/adhoc_import:binary_import_flags",
        "//learning/deepmind/python/adhoc_import:colab_import",
        "//learning/deepmind/python/ipython",
        "//learning/deepmind/tensorflow/sstable",
        "//learning/deepmind/xmanager2/client/google",
        "//pyglib:build_data",
        "//pyglib:gfile",
        "//pyglib:resources",
        "//pyglib/contrib/gpathlib:gpath",
        "//research/drishti/app/pursuit/data/proto:annotation_proto_py_pb2",
        "//research/drishti/app/pursuit/data/proto:object_proto_py_pb2",
        "//research/drishti/app/pursuit/data/schema:features",
        "//research/vision/scene_understanding/data/tfds:example_parser",
        "//research/vision/scene_understanding/data/tfds:image_features",
        "//research/vision/scene_understanding/datasets/shapenet:shapenet_specs",
        "//research/vision/scene_understanding/datasets/shapenet:shapenet_utils",
        "//research/vision/scene_understanding/isun",
        "//research/xeno/nnets/common:files",
        "//third_party/open3d",
        "//third_party/py/PIL:pil",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/chex",
        "//third_party/py/colorama",
        "//third_party/py/cvx2",
        "//third_party/py/flax",
        "//third_party/py/flax/metrics:tensorboard",
        "//third_party/py/flax/training",
        "//third_party/py/imageio",
        "//third_party/py/jax",
        "//third_party/py/jax/jaxlib",
        "//third_party/py/jax/jaxlib:cpu_feature_guard",
        "//third_party/py/mediapy",
        "//third_party/py/numpy",
        "//third_party/py/plotly",
        "//third_party/py/plotly:plotly_colab",
        "//third_party/py/skimage",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow:tensorflow_google",
        "//third_party/py/tensorflow_hub",
        "//third_party/py/yaml",
        "//third_party/tensorboard/plugins/mesh",
        "//third_party/tensorflow/python/tools:freeze_graph_lib",
        "//third_party/tensorflow_models/object_detection/utils:visualization_utils",
    ],
)
