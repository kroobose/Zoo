python test_lfw.py \
    --test_set 'RFW_Indian' \
    --data_conf_file 'data_conf_test.yaml' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file 'backbone_conf.yaml' \
    --batch_size 2048 \
    --model_path '/workspace/training_mode/conventional_training/out_dir'
