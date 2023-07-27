GPU_NUM=7
TRAIN_CONFIG_YAML="configs/base_modl,k=10.yaml"

CUDA_VISIBLE_DEVICES=$GPU_NUM python train.py \
    --config=$TRAIN_CONFIG_YAML \
    --write_image=1