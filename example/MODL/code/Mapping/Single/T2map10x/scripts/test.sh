GPU_NUM=4
TEST_CONFIG_YAML="configs/base_modl,k=10.yaml"

CUDA_VISIBLE_DEVICES=$GPU_NUM python test.py \
    --config=$TEST_CONFIG_YAML \
    --batch_size=8 \
    --write_image=1