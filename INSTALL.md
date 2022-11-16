
## Installation
please check [INSTALL.md](INSTALL.md) for installation instructions

## Generation data

 
1. Cover the radar pcd file as image
```shell
cd your_repositroy
python tool/waterScene/convert_radar_point.py --dataroot your_dataset_location
```

2 generate the norm infor of radar images.
```shell
python tool/waterScene/generate_2d_coco_pc_norm_info.py --dataroot your_dataset_location
```

## Training
```shell
python -m torch.distributed.launch \
       --nproc_per_node=your_gpu_numbers \
       --master_port=$((RANDOM + 10000)) \
       tools/train_net.py \
       --config-file your_config_file_location \
       OUTPUT_DIR your_output_location
       
```

## Testing
```shell
python -m torch.distributed.launch \
       --nproc_per_node=your_gpu_numbers \
       --master_port=$((RANDOM + 10000)) \
       tools/test_net.py \
       --config-file your_config_file_location \
       OUTPUT_DIR your_output_location
       
```

## Result Illustration
```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.525
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.866
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.560
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.327
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.521
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.586
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.274
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.613
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.628
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.409
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.624
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.684
