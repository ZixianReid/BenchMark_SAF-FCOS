python -m torch.distributed.launch \
       --nproc_per_node=1 \
       --master_port=$((RANDOM + 10000)) \
       tools/train_net.py