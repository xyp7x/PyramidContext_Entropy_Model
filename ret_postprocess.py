import os
import numpy as np 
import cv2 
from utils.codec import filesize
import pandas as pd
import argparse

def merge_multi_output(pre_results_folder):
    print(pre_results_folder)
    # select the output file
    output_list = []
    file_list = os.listdir(pre_results_folder)
    file_list.sort()
    for file in file_list:
        if ('_output_' in file):
            print(file)
            output_list.append(file)
    output_list.sort()
    
    # read the pre results
    print('read!')
    merge_results = []
    for output_file in output_list:
        output_file_path = os.path.join(pre_results_folder, output_file)
        results_num = int(output_file[:-4].split('_')[-1])
        with open(output_file_path, 'r') as f:
            results_line = f.readlines()
            if results_num == 0:
               header = results_line[0]
            for line in results_line[1:]:
                if 'ImageID' in line:
                    continue
                if 'total_file_size' in line:
                    continue
                merge_results.append(line)
    merge_results.sort()
    print(len(merge_results))
    
    # write to a new file
    print('write!')
    merge_file = os.path.join(pre_results_folder, './output_all.txt')
    with open(merge_file, 'w') as f:
        f.write(header)
        for result in merge_results:
            f.write(result)        

def merge_multi_output_folder(pre_results_folder_multi):
    folder_list = os.listdir(pre_results_folder_multi)
    folder_list.sort()
    for folder in folder_list:
        if 'lambda' in folder:
            pre_results_folder = os.path.join(pre_results_folder_multi, folder)
            if len(os.listdir(pre_results_folder)) == 0:
                continue
            merge_multi_output(pre_results_folder)

def cal_bpp():
    img_folder_path = '/apdcephfs_cq2/share_1501061/Datasets/openImageV6/val'
    cf_features_save_path = '/apdcephfs_cq2/share_1501061/person/zy/VCM_test/p1/compress_test_allP_check/compress_file_fromimg/cf_check02_unify_pretrained_model_249_downsamplep2' 
    
    file_list = os.listdir(cf_features_save_path)
    file_list.sort()
    for file in file_list:
        if 'lambda' not in file:
            continue
        file_folder_path = os.path.join(cf_features_save_path, file)
        features_list = os.listdir(file_folder_path)
        features_list.sort()
        total_pixel_num = 0
        total_file_size = 0
        total_img_size = 0

        for feature_file in features_list:
            # pixel num
            img_name = feature_file + '.jpg'
            img_path = os.path.join(img_folder_path, img_name)
            img = cv2.imread(img_path)
            pixel_num = img.shape[0]*img.shape[1]

            total_pixel_num += pixel_num
            total_img_size +=float(filesize(img_path))
            
            # file size
            features_path = os.path.join(file_folder_path, feature_file)
            layer_file_list = os.listdir(features_path)
            for layer_file in layer_file_list:
                layer_file_path = os.path.join(features_path, layer_file)
                file_size =  float(filesize(layer_file_path)) 
                total_file_size += file_size
        
        bpp = (total_file_size*8)/total_pixel_num
        print(file, 'bpp:', bpp, 'total_file_size', total_file_size, 'total_pixel_num', total_pixel_num, 'total_img_size:', total_img_size)

def cal_bpp_from_log(pre_results_folder):
    pre_file_list = os.listdir(pre_results_folder)
    pre_file_list.sort()
    bpps = []
    for pre_file in pre_file_list:
        pre_results_path = os.path.join(pre_results_folder, pre_file)
        file_list = os.listdir(pre_results_path)
        file_list = list(filter(lambda file: 'TestImgLog' in file, file_list))
        file_list.sort()
        file_size = 0
        pixel_num = 0
        TestImgLog_all=dict()
        for file in file_list:
            file_size += int(file.split('_')[3])
            pixel_num += int(file.split('_')[2])
            
            with open(os.path.join(pre_results_folder,pre_file,file),'r') as sep_f:
                lines = sep_f.readlines() 
                for line in lines:   
                    line=line.strip().split(':')
                    if line[0] not in TestImgLog_all:
                        TestImgLog_all[line[0]]=float(line[1])
                    else:
                        TestImgLog_all[line[0]]+=float(line[1])

        if pixel_num ==0:
            bpp=0
        else:
            bpp = float(file_size) *8/pixel_num
        TestImgLog_all['bpp']=bpp

        with open(os.path.join(pre_results_folder,pre_file,'TestImgAll.txt'),'w') as all_f:
            for key,value in TestImgLog_all.items():
                all_f.write('{0}:{1}\n'.format(key,value))
                
        bpps.append(bpp)

    return bpps    

def cvt_detectron_coco_oid(coco_output_folder, selected_classes):
    # selected_classes = '/apdcephfs_cq2/share_1501061/person/zy/VCM_test/p1/compress_test_allP_check/detectron2/data/selected_classes.txt'
    file_list = os.listdir(coco_output_folder)
    file_list.sort()
    
    # unify class name by replacing space to underscore
    def unify_name(class_name):
        return class_name.replace(' ', '_')
    
    # selected coco classes
    selected_coco_classes_fname = selected_classes
    with open(selected_coco_classes_fname, 'r') as f:
        selected_classes = [unify_name(x) for x in f.read().splitlines()]
    
    for file in file_list:
        if 'lambda' not in file:
            continue
        if len(os.listdir(os.path.join(coco_output_folder, file))) == 0:
            continue
        coco_output_file = os.path.join(os.path.join(coco_output_folder, file), 'output_all.txt')
        print(coco_output_file)
        oid_output_file = coco_output_file.replace('.txt', '_oid.txt')
        
        # prediciton output
        coco_fname = coco_output_file
        oid_fname = oid_output_file
        
        # load coco output data file
        coco_output_data = pd.read_csv(coco_fname)

        # generate output
        of = open(oid_fname, 'w')

        #write header
        of.write(','.join(coco_output_data.columns)+'\n')

        # iterate all input files
        for idx, row in coco_output_data.iterrows():
            fields = row.tolist()
            coco_id = unify_name(row['LabelName'])
            if coco_id in selected_classes:
                oid_id = coco_id
                row['LabelName'] = oid_id
                o_line = ','.join(map(str,row))
                of.write(o_line + '\n')

        of.close()


if __name__ == '__main__':
    # pre_results_folder = '/apdcephfs_cq2/share_1501061/person/zy/VCM_test/p1/compress_test_allP_check/pre_results_detection/pre_check_unify_pretrained_model_200_downsamplep2'
    # selected_classes = '/apdcephfs_cq2/share_1501061/person/zy/VCM_test/p1/compress_test_allP_check/detectron2/data/selected_classes.txt'
    # merge_multi_output(pre_results_folder)
    # cal_bpp()
    
    parser = argparse.ArgumentParser(description="postprocess the predicted results .")
    parser.add_argument(
        "--pre_results_folder",
        type=str,
        default="/home/dingding/MM/VCM/FCVCM/CfP/compress_test_allP_check/pre_results_detection/pre_2.0_downsamplep2",
        help="the folder of predicted results",
    )
    parser.add_argument(
        "--selected_classes",
        type=str,
        default="/home/dingding/MM/VCM/FCVCM/CfP/compress_test_allP_check/detectron2/data/selected_classes.txt",
        help="used for cvt class id",
    )
    args = parser.parse_args()
    
    merge_multi_output_folder(args.pre_results_folder)
    cvt_detectron_coco_oid(args.pre_results_folder, args.selected_classes)
    cal_bpp_from_log(args.pre_results_folder)