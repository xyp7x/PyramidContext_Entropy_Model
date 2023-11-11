# Copyright (c) 2020-2021 Nokia Corporation and/or its subsidiary(-ies). All rights reserved.
# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

# evaluation according to instruction from 
#https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/challenge_evaluation.md#object-detection-track

set -e

# load configuration
file_folder=$(realpath /home/dingding/MM/VCM/FCVCM/CfP/DKIC_CfP/)
dataset_dir=$(realpath /home/dingding/Dataset/VCM/)
selected_classes_path=$file_folder/detectron2/data/selected_classes.txt

# modified the path of predicted results
pre_results_root_dir=$file_folder/pre_results_segmentation
pre_result_dir=$pre_results_root_dir/pre_opimg_seg_unify_pretrained_model_350_downsamplep2

# load the file for evaluation
anno_dir=$dataset_dir/annotations_5k
segmentation_mask_dir=$anno_dir/challenge_2019_validation_masks
BOUNDING_BOXES=$anno_dir/segmentation_validation_bbox_5k.csv
IMAGE_LABELS=$anno_dir/segmentation_validation_labels_5k.csv
INSTANCE_SEGMENTATIONS=$anno_dir/segmentation_validation_masks_5k.csv
LABEL_MAP=$anno_dir/coco_label_map.pbtxt

# post the predicted results, so that it could be evaluated by the configured API 
python $file_folder/PostprocessResults.py \
    --pre_results_folder=${pre_result_dir} \
    --selected_classes=${selected_classes_path}

# evaluation the predicted results
# get TF object_detection path
TF_od_dir=$(python -c "import object_detection,os;print(os.path.dirname(object_detection.__file__))")

# evaluation
# for lambda in 0.016 0.032 0.064 0.128 0.256 0.512 1.0 2.0; do
for lambda in 0.18 0.36; do
    echo processing ...$lambda 

    INPUT_PREDICTIONS=$pre_result_dir/lambda${lambda}/output_all_oid.txt
    OUTPUT_METRICS=$pre_result_dir/lambda${lambda}/output_all_oid_metric.txt
    echo $INPUT_PREDICTIONS

    python -u $TF_od_dir/metrics/oid_challenge_evaluation.py \
        --input_annotations_boxes=${BOUNDING_BOXES} \
        --input_annotations_labels=${IMAGE_LABELS} \
        --input_class_labelmap=${LABEL_MAP} \
        --input_annotations_segm=${INSTANCE_SEGMENTATIONS} \
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
