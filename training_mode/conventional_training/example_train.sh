mkdir 'log'
python train.py \
    --data_root '/workspace/dataset/DigiFace-1M/5_images_per_identity' \
    --train_file '/workspace/dataset/DigiFace-1M/5_images_per_identity/train_list.txt' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../backbone_conf.yaml' \
    --head_type 'MagFace' \
    --head_conf_file '../head_conf_digi5.yaml' \
    --lr 0.1 \
    --out_dir 'out_dir_5imgs_resnet' \
    --epoches 50 \
    --step '10, 20, 30, 40, 50' \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 256 \
    --momentum 0.9 \
    --log_dir 'log' \
    --tensorboardx_logdir 'mv-hrnet' \
    2>&1 | tee log/log.log

