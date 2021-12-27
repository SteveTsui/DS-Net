python -m torch.distributed.launch --master_port 3432 --nproc_per_node=8 --use_env main.py \
--model ds_net_tiny/ds_net_small --batch-size 128 --data-path path_to_dataset --epochs 300 \
--output_dir work_dirs