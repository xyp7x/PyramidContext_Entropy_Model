
datasetdir=$(realpath /home/dingding/workspace/MM/Dataset/FCM/)
projectdir=$(realpath /home/dingding/workspace/MM/FCM/PyramidContext_Entropy_Model/)

task=opimg_det
quality_level1=10

CUDA_VISIBLE_DEVICES=4 python3 -u train.py --task_extractor ${task} -m JointAutoregressiveHierarchicalPriors_Channel256 --dataloader_type FeaturesFromImg_SingleLayer -d $datasetdir --train_dataset train50k --test_dataset ${task}_test --quality-level ${quality_level1} -e 300 -n 1 --batch-size 8 --test-batch-size 8 --learning-rate 1e-4 --aux-learning-rate 1e-3 --patch-size 256 256 --cuda --save --clip_max_norm 1.0 --save_location $projectdir/saveModels --model_pretrained_path $projectdir/detectron2_cfg/pretrained_models
# quality_level = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12




