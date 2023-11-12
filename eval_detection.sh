# Copyright (c) 2020-2021 Nokia Corporation and/or its subsidiary(-ies). All rights reserved.
# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

# evaluation according to instruction from 
#https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/challenge_evaluation.md#object-detection-track

set -e

# load configuration
file_folder=$(realpath /home/dingding/workspace/MM/FCM/PyramidContext_Entropy_Model/)
dataset_dir=$(realpath /home/dingding/workspace/MM/Dataset/FCM/)

# modified the path of predicted results
pre_results_root_dir=$file_folder/ret_detection
pre_result_dir=$pre_results_root_dir/ret_opimg_det_pretrained_model_100

# load the file for evaluation
anno_dir=$dataset_dir/annotations_5k
selected_classes_path=$anno_dir/selected_classes.txt
BOUNDING_BOXES=$anno_dir/detection_validation_5k_bbox.csv
IMAGE_LABELS=$anno_dir/detection_validation_labels_5k.csv
LABEL_MAP=$anno_dir/coco_label_map.pbtxt

# post the predicted results, so that it could be evaluated by the configured API 
python $file_folder/ret_postprocess.py \
    --pre_results_folder=${pre_result_dir} \
    --selected_classes=${selected_classes_path}

# evaluation the predicted results
# get TF object_detection path
TF_od_dir=$(python -c "import object_detection,os;print(os.path.dirname(object_detection.__file__))")

# evaluation
# for lambda in 0.016 0.032 0.064 0.128 0.256 0.512 1.0 2.0; do
for lambda in 0.0932 0.72; do
    echo processing ...$lambda 

    INPUT_PREDICTIONS=$pre_result_dir/lambda${lambda}/output_all_oid.txt
    OUTPUT_METRICS=$pre_result_dir/lambda${lambda}/output_all_oid_metric.txt
    echo $INPUT_PREDICTIONS

    python -u $TF_od_dir/metrics/oid_challenge_evaluation.py \
        --input_annotations_boxes=${BOUNDING_BOXES} \
        --input_annotations_labels=${IMAGE_LABELS} \
        --input_class_labelmap=${LABEL_MAP} \
        --input_predictions=${INPUT_PREDICTIONS} \
        --output_metrics=${OUTPUT_METRICS} 

    echo 
    echo 'Evaluation results: '
    echo 

    OOUTPUT_METRICS=$pre_result_dir/lambda${lambda}/output_all_oid_metric.txt
    rslt=$(head $OUTPUT_METRICS -n 1 | cut -d , -f 2)
    echo $rslt
done

echo Done!