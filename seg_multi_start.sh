model_num_min=0
model_num_max=2
file_dir=$(realpath /home/dingding/MM/VCM/FCVCM/CfP/DKIC_CfP/) 

echo 'compress and decompress begin!'
CUDA_VISIBLE_DEVICES=1 python3  $file_dir/decompress_predictor_FromImg256_downsamplep2.py  \
        --checkpoint_folder_path  $file_dir/saveModels/opimg_seg_unify_pretrained_model_350 \
        --img_folder_path /home/dingding/Dataset/VCM/opimg_seg_test \
        --img_test_file /home/dingding/Dataset/VCM/annotations_5k/segmentation_validation_input_5k.lst \
        --cf_save_folder_path $file_dir/compress_file_fromimg \
        --pre_save_folder_path $file_dir/pre_results_segmentation \
        --cococlass_path $file_dir/detectron2/data/coco_classes.txt \
        --model_ar DKIC_features_input256 --model_channels 256 \
        --task_predictor segmentation --task_extractor segmentation --device_num 0 \
        --test_model_num_min $model_num_min --test_model_num_max $model_num_max \
        --results_num 0 --test_img_num_min 0 --test_img_num_max 2500 \
& \
CUDA_VISIBLE_DEVICES=1 python3  $file_dir/decompress_predictor_FromImg256_downsamplep2.py  \
        --checkpoint_folder_path  $file_dir/saveModels/opimg_seg_unify_pretrained_model_350 \
        --img_folder_path /home/dingding/Dataset/VCM/opimg_seg_test \
        --img_test_file /home/dingding/Dataset/VCM/annotations_5k/segmentation_validation_input_5k.lst \
        --cf_save_folder_path $file_dir/compress_file_fromimg \
        --pre_save_folder_path $file_dir/pre_results_segmentation \
        --cococlass_path $file_dir/detectron2/data/coco_classes.txt \
        --model_ar DKIC_features_input256 --model_channels 256 \
        --task_predictor segmentation --task_extractor segmentation --device_num 0 \
        --test_model_num_min $model_num_min --test_model_num_max $model_num_max \
        --results_num 1 --test_img_num_min 2500 --test_img_num_max 5000 \

echo 'compress and decompress finished!'
