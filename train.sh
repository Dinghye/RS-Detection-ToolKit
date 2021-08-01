# Linux 下换行符为 CRLF 的需改为 LF
# lr = 0.00025 * num_gpus
python3 train_net.py \
  --config-file model/my_config.yaml \
  --num-gpus 1 \
  SOLVER.IMS_PER_BATCH 12 \
  SOLVER.BASE_LR 0.00025 \
  SOLVER.MAX_ITER 8000 \
  SOLVER.STEPS '(2400, 2900)'
