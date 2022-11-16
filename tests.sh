#python -m torch.distributed.launch \
#       --nproc_per_node=2 \
#       --master_port=$((RANDOM + 10000)) \
#       tools/test_net.py \
#       --config-file /ZXH_Zixian_Zhang/code/FCOS/configs/fcos/fcos_imprv_R_50_FPN_1x.yaml \
#       --checkpointer_file tmp/model_final.pth \
#       OUTPUT_DIR tmp/fcos_imprv_R_50_FPN_1x_ADD_Circle_07

python tools/test_net.py \
    --config-file /ZXH_Zixian_Zhang/code/SAF-FCOS/configs/WaterScene/fcos_imprv_R_50_FPN_FUSION.yaml \
    MODEL.WEIGHT tmp/model_final.pth \
    TEST.IMS_PER_BATCH 4