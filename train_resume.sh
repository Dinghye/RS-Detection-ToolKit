# 断点续 train
# --num-gpus 亲测不能省略
python3 train_net.py \
  --config-file model/my_config.yaml \
  --num-gpus 1 \
  --resume
