model_num_min=1
model_num_max=2
file_dir=$(realpath /home/dingding/workspace/MM/FCM/ScaleA_Entropy_Model) 
dataset_dir=$(realpath /home/dingding/workspace/MM/Dataset/FCM/)

echo 'compress and decompress begin!'
CUDA_VISIBLE_DEVICES=0 python3 $file_dir/inference.py  \
        --checkpoint_folder_path  $file_dir/saveModels/opimg_det_pretrained_model_100 \
        --img_folder_path $dataset_dir/opimg_det_test \
        --img_test_file $dataset_dir/annotations_5k/detection_validation_input_5k.lst \
        --bs_save_folder_path $file_dir/bistream \
        --fd_save_folder_path $file_dir/feature_dumps \
        --ret_save_folder_path $file_dir/ret_detection \
        --cococlass_path $dataset_dir/annotations_5k/coco_classes.txt \
        --model JointAutoregressiveHierarchicalPriors_Channel256 \
        --task_nn detection --device_num 0 \
        --p2_downsampling_ratio 1 --p3_downsampling_ratio 1 \
        --test_model_num_min $model_num_min --test_model_num_max $model_num_max \
        --results_num 0 --test_img_num_min 0 --test_img_num_max 2500 \
& \
CUDA_VISIBLE_DEVICES=1 python3 $file_dir/inference.py  \
        --checkpoint_folder_path  $file_dir/saveModels/opimg_det_pretrained_model_100 \
        --img_folder_path $dataset_dir/opimg_det_test \
        --img_test_file $dataset_dir/annotations_5k/detection_validation_input_5k.lst \
        --bs_save_folder_path $file_dir/bistream \
        --fd_save_folder_path $file_dir/feature_dumps \
        --ret_save_folder_path $file_dir/ret_detection \
        --cococlass_path $dataset_dir/annotations_5k/coco_classes.txt \
        --model JointAutoregressiveHierarchicalPriors_Channel256 \
        --task_nn detection --device_num 0 \
        --p2_downsampling_ratio 1 --p3_downsampling_ratio 1 \
        --test_model_num_min $model_num_min --test_model_num_max $model_num_max \
        --results_num 1 --test_img_num_min 2500 --test_img_num_max 5000 \


echo 'compress and decompress finished!'
