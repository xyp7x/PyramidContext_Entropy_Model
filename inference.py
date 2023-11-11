# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Evaluate an end-to-end compression model on an image dataset.
"""

import os
import sys

#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
current_dir=os.path.dirname(os.path.abspath(__file__))


import time
from pathlib import Path
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2 import model_zoo
from detectron2.config import get_cfg
from dataset.detectron2_feature_NN import PlayerPredictor
from utils import oid_mask_encoding
from utils.codec import filesize, write_body, read_body, write_uints, read_uints
from utils.feature_dump import FeatureDump

from models.sa_entropy_model import JointAutoregressiveHierarchicalPriors_Channel256


# add model
model_feature_ar = {}
model_feature_ar['JointAutoregressiveHierarchicalPriors_Channel256'] = JointAutoregressiveHierarchicalPriors_Channel256


def extend(x, base_num):
    _, _, h, w = x.size()
    ex_h = 0
    ex_w = 0
    out = x
    if h % base_num != 0:
        ex_h = base_num - h % base_num
    if w % base_num != 0:
        ex_w = base_num - w % base_num
    out = nn.functional.pad(x,
                            pad=(0, ex_w, 0, ex_h),
                            mode="constant",
                            value=0)
    return out, ex_h, ex_w


# record the info of test
def find_testimg_num(args, results_path):
    file_list = os.listdir(results_path)
    test_img_num = -1

    for file in file_list:
        if 'TestImgLog_{}'.format(args.results_num) in file:
            test_img_num = int(file.split('.')[0].split('_')[-1])
            count_pixel_num = int(file.split('.')[0].split('_')[2])
            count_filesize = int(file.split('.')[0].split('_')[3])
            log_name = os.path.join(results_path, file)
            args.test_img_num_min = test_img_num
            args.count_pixel_num = count_pixel_num
            args.count_filesize = count_filesize
            break

    if test_img_num == -1:
        log_name = os.path.join(
            results_path,
            'TestImgLog_{0}_{1}_{2}_{3}.txt'.format(args.results_num,
                                                    args.count_pixel_num,
                                                    args.count_filesize,
                                                    args.test_img_num_min))
        f = open(log_name, 'w')
        f.close()

    return log_name


# encode the feature data of one P layer
def encode_feature(model,
                   x,
                   input,
                   f,
                   enc_image_size=False,
                   enc_p2_size=False):

    with torch.no_grad():
        starttime = time.time()
        out = model.compress(x)
        
        fea_encode_time = time.time() - starttime

    # write original image size
    if enc_image_size:
        write_uints(f, input['image_size'])

    # write p2 size
    if enc_p2_size:
        write_uints(f, input['p2_size'])

    # write the range values for reversing the normalization
    write_uints(f, (input['min'] + 100, input['max'] +
                    100))  #### make sure the  values to be positive

    # write shape and number of encoded latents
    write_body(f, out["shape"], out["strings"])

    return fea_encode_time


# encode the feature data of all P layers from one image
def encode_feature_file(model,
                        extractor,
                        p2_downsampling_ratio,
                        p3_downsampling_ratio,
                        img_path,
                        compress_file_save,
                        device,
                        base_num=16):
    img = cv2.imread(img_path)
    img_size = (img.shape[0], img.shape[1])
    img_name = os.path.basename(img_path)
    bs_img_path = os.path.join(compress_file_save, img_name.split('.')[0] + '.bin')
    if os.path.exists(bs_img_path):
        os.remove(bs_img_path)

    starttime = time.time()
    features = extractor(img, save_player=True)
    img_part1_time = time.time() - starttime

    pixel_num = img.shape[0] * img.shape[1]

    # compress features
    img_conversion_time = 0
    img_encode_time = 0

    with Path(bs_img_path).open("wb") as f:
        for key, values in features.items():

            starttime = time.time()

            if key == 'p6':
                continue

            enc_image_size = False
            enc_p2_size = False
            input = {}
            if key == 'p2':
                input['image_size'] = img_size
                input['p2_size'] = (values.size()[-2], values.size()[-1])
                enc_image_size = True
                enc_p2_size = True
                if p2_downsampling_ratio>0:
                    values = F.interpolate(values, scale_factor=p2_downsampling_ratio,
                                    mode='bicubic', recompute_scale_factor=True, align_corners = True)  # downsampling p2 layer
            elif key == 'p3':
                if p3_downsampling_ratio>0:
                    values = F.interpolate(values, scale_factor=p3_downsampling_ratio,
                                    mode='bicubic', recompute_scale_factor=True, align_corners = True)  # downsampling p3 layer

            # pre-process of features
            input['features'] = (extend(values, base_num)[0]).to(device)
            input['max'] = int(np.ceil(torch.max(values).item()))
            input['min'] = int(np.floor(torch.min(values).item()))
            # normalize to [0, 1]
            x = (input['features'] - input['min']) / (input['max'] - input['min'])

            img_conversion_time += time.time() - starttime

            fea_encode_time = encode_feature(
                model, x, input, f, enc_image_size, enc_p2_size)
            img_encode_time += fea_encode_time

    file_size=filesize(bs_img_path)


    return pixel_num, file_size, img_part1_time, img_conversion_time, img_encode_time


# decode the compressed feature data of all P layers
def decode_feature_file(model, p2_downsampling_ratio, p3_downsampling_ratio,compress_features_path,img_name):
    decompress_feature = {}
    layer_names=['p2','p3','p4','p5']
    bs_img_path=os.path.join(compress_features_path, img_name.split('.')[0] + '.bin')

    file_size = filesize(bs_img_path)
    pixel_num = 0
    img_decode_time = 0
    img_inv_conversion_time = 0
    
    with Path(bs_img_path).open("rb") as f:

        for layer_name in layer_names:
            dec_image_size = False
            dec_p2_size = False
            if layer_name == 'p2':
                dec_image_size = True
                dec_p2_size = True

            # decompress file
            starttime = time.time()
            
            if dec_image_size:
                img_size = read_uints(f, 2)
                image_size=img_size
            if dec_p2_size:
                p2_size = read_uints(f, 2)

            min_values, max_values = read_uints(f, 2)
            min_values = min_values - 100
            max_values = max_values - 100

            strings, shape = read_body(f)
            with torch.no_grad():
                out = model.decompress(strings, shape)

            img_decode_time += time.time() - starttime

            # post-process of the decompressing features
            starttime = time.time()
            h_p2 = p2_size[0]
            w_p2 = p2_size[1]
            layer_num = int(layer_name[1])
            h_pi = np.ceil(h_p2 / (2**(layer_num - 2))).astype(np.int32)
            w_pi = np.ceil(w_p2 / (2**(layer_num - 2))).astype(np.int32)

            if layer_name == 'p2' and p2_downsampling_ratio>0:
                out_merge = out['x_hat'][:, :, :int(h_pi*p2_downsampling_ratio), :int(w_pi *p2_downsampling_ratio)]
                out_merge = F.interpolate(
                    out_merge, size=p2_size,
                    mode='bicubic',align_corners = True)  # upsampling p2 layer to original size
            elif layer_name == 'p3' and p3_downsampling_ratio>0:
                out_merge = out['x_hat'][:, :, :int(h_pi*p3_downsampling_ratio), :int(w_pi*p3_downsampling_ratio)]
                out_merge = F.interpolate(
                    out_merge, size=(h_pi,w_pi),
                    mode='bicubic',align_corners = True)  # upsampling p2 layer to original size
            else:
                out_merge = out['x_hat'][:, :, :h_pi, :w_pi]
            decompress_feature[layer_name] = out_merge * (max_values -
                                                        min_values) + min_values
            img_inv_conversion_time += time.time() - starttime

    pixel_num = image_size[0] * image_size[1]
    assert pixel_num>0
    
    starttime = time.time()
    decompress_feature['p6'] = F.max_pool2d(decompress_feature['p5'],
                                            kernel_size=1,
                                            stride=2,
                                            padding=0)
    img_inv_conversion_time += time.time() - starttime

    return decompress_feature, pixel_num, file_size, img_decode_time, img_inv_conversion_time


def encode_decode_and_predict(args, model, task_nn,
                              bs_save_lambda_path, fd_save_lambda_path, file_ret_save_path,
                              img_list, of, coco_classes, log_name, device):

    total_pixel_num = args.count_pixel_num
    total_file_size = args.count_filesize

    print('the num of test images, from {0} to {1}, in results {2}'.format(
        args.test_img_num_min, args.test_img_num_max, args.results_num))

    fd = FeatureDump()
    layer_nb=4
    layer_channel_nb=256
    for img_coded_idx in range(args.test_img_num_min):
        for layer_idx in range(layer_nb):
            # calculating the remainder of the offset
            fd._offset = (fd._offset + layer_channel_nb) % fd._subsample

    part1_time = 0
    conversion_time = 0
    encode_time = 0
    decode_time = 0
    inv_conversion_time = 0
    part2_time = 0
    for i in range(args.test_img_num_min, args.test_img_num_max):
        # read image
        img_name = img_list[i].replace('png', 'jpg')
        img_path = os.path.join(args.img_folder_path, img_name)
        img = cv2.imread(img_path)
        print(f'{args.results_num} processing {i}:{img_name}...')

        # compress the features of img
        pixel_num, file_size, img_part1_time, img_conversion_time, img_encode_time = encode_feature_file(
            model,
            task_nn,
            args.p2_downsampling_ratio,
            args.p3_downsampling_ratio,
            img_path,
            bs_save_lambda_path,
            device,
            base_num=64)
        total_pixel_num += pixel_num
        total_file_size += file_size
        part1_time += img_part1_time
        conversion_time += img_conversion_time
        encode_time += img_encode_time

        # load feature and decompress
        decompress_feature, pixel_num, file_size, img_decode_time, img_inv_conversion_time = decode_feature_file(
            model, args.p2_downsampling_ratio,args.p3_downsampling_ratio, bs_save_lambda_path, img_name)
        decode_time += img_decode_time
        inv_conversion_time += img_inv_conversion_time
        total_pixel_num += pixel_num
        total_file_size += file_size

        feature_dump_layers=[decompress_feature[layer_name].cpu().numpy() for layer_name in ['p2','p3','p4','p5']]
        fd.set_fptr(open(os.path.join(fd_save_lambda_path,img_name.replace('.jpg','.dump')),'w'))
        fd.write_layers(feature_dump_layers)  # tensor_list is a list of BCHW tensors
        
        # using decompress feature to predict
        starttime = time.time()
        outputs = task_nn(original_image=img,
                            player=decompress_feature,
                            using_saved_player=True)[0]

        # process the results
        classes = outputs['instances'].pred_classes.to('cpu').numpy()
        scores = outputs['instances'].scores.to('cpu').numpy()
        bboxes = outputs['instances'].pred_boxes.tensor.to('cpu').numpy()
        H, W = outputs['instances'].image_size
        # convert bboxes to 0-1
        bboxes = bboxes / [W, H, W, H]
        # detectron: x1, y1, x2, y2 in pixels
        # OpenImage output x1, x2, y1, y2 in percentage
        bboxes = bboxes[:, [0, 2, 1, 3]]

        if args.task_nn == 'segmentation':
            masks = outputs['instances'].pred_masks.to('cpu').numpy()

        for ii in range(len(classes)):
            coco_cnt_id = classes[ii]
            class_name = coco_classes[coco_cnt_id]

            if args.task_nn == 'segmentation':
                assert (masks[ii].shape[1] == W) and (
                    masks[ii].shape[0] == H
                ), print(
                    'Detected result does not match the input image size: ',
                    img_name[:-4])

            rslt = [img_name[:-4], class_name, scores[ii]
                    ] + bboxes[ii].tolist()

            if args.task_nn == 'segmentation':
                rslt += [
                    masks[ii].shape[1], masks[ii].shape[0],
                    oid_mask_encoding.encode_binary_mask(
                        masks[ii]).decode('ascii')
                ]

            o_line = ','.join(map(str, rslt))
            of.write(o_line + '\n')
            of.flush()
        part2_time += time.time() - starttime

        #record tested image
        log_undate_name = os.path.join(
            file_ret_save_path,
            'TestImgLog_{0}_{1}_{2}_{3}.txt'.format(args.results_num,
                                                    total_pixel_num,
                                                    total_file_size, i + 1))
        os.rename(log_name, log_undate_name)
        log_name = log_undate_name

    bpp = total_file_size * 8 / total_pixel_num
    print(
        'total_file_size:{0}\n total_pixel_num:{1}\n bpp:{2}\n nn_part1_time:{3}\n conversion_time:{4}\n encode_time:{5}\n decode_time:{6}\n inv_conversion_time:{7}\n nn_part2_time:{8}'
        .format(total_file_size, total_pixel_num, bpp, part1_time,
                conversion_time, encode_time, decode_time, inv_conversion_time,
                part2_time))
    f_log = open(log_name, 'w')
    f_log.write(
        'total_file_size:{0}\n total_pixel_num:{1}\n bpp:{2}\n nn_part1_time:{3}\n conversion_time:{4}\n encode_time:{5}\n decode_time:{6}\n inv_conversion_time:{7}\n nn_part2_time:{8}'
        .format(total_file_size, total_pixel_num, bpp, part1_time,
                conversion_time, encode_time, decode_time, inv_conversion_time,
                part2_time))
    f_log.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compressing script.")
    parser.add_argument(
        "-m",
        "--model",
        default="JointAutoregressiveHierarchicalPriors_Channel256",
        help="Model architecture",
    )
    parser.add_argument(
        "--p2_downsampling_ratio",
        type=float,
        default=-1,
        help="the downsampling ratio of p2 features",
    )
    parser.add_argument(
        "--p3_downsampling_ratio",
        type=float,
        default=-1,
        help="the downsampling ratio of p3 features",
    )
    parser.add_argument(
        "--checkpoint_folder_path",
        default=os.path.join(current_dir,"saveModels/opimg_det_pretrained_model_30"),
        help="the folder of pretrained models needed to test")

    parser.add_argument(
        "--img_folder_path",
        type=str,
        default='/home/dingding/workspace/MM/Dataset/FCM/opimg_det_test',
        help="the folder of test image, using to compute the size of image")

    parser.add_argument(
        "--img_test_file",
        type=str,
        default='/home/dingding/workspace/MM/Dataset/FCM/annotations_5k/detection_validation_input_5k.lst',
        help="the image list for test")

    parser.add_argument("--bs_save_folder_path",
                        type=str,
                        default=os.path.join(current_dir,'bistream'),
                        help="the folder of saving compressed file")

    parser.add_argument("--fd_save_folder_path",
                        type=str,
                        default=os.path.join(current_dir,'feature_dumps'),
                        help="the folder of feature dumps")

    parser.add_argument("--ret_save_folder_path",
                        type=str,
                        default=os.path.join(current_dir, 'ret_detection'),
                        help="the folder of saving predicted file")

    parser.add_argument("--cococlass_path",
                        type=str,
                        default='/home/dingding/workspace/MM/Dataset/FCM/annotations_5k/coco_classes.txt',
                        help="the path of cococlasss")

    parser.add_argument("--task_nn",
                        type=str,
                        default='detection',
                        help="the task to extract and predict the features")

    parser.add_argument("--device_num",
                        type=str,
                        default='0',
                        help="the num of used gpu")

    parser.add_argument("--results_num",
                        type=str,
                        default='0',
                        help="the num of test multiprocess")

    parser.add_argument("--test_model_num_min",
                        type=int,
                        default=0,
                        help="the min num of test model")

    parser.add_argument("--test_model_num_max",
                        type=int,
                        default=1,
                        help="the max num of test model")

    parser.add_argument("--test_img_num_min",
                        type=int,
                        default=0,
                        help="the min num of test model")

    parser.add_argument("--test_img_num_max",
                        type=int,
                        default=5000,
                        help="the max num of test model")

    parser.add_argument("--count_pixel_num",
                        type=int,
                        default='0',
                        help="the num of pixels of processed images")

    parser.add_argument("--count_filesize",
                        type=int,
                        default='0',
                        help="the filesize of compressed features")

    args = parser.parse_args()

    device = "cuda"

    # creat the folder to save the compressing features
    if not os.path.exists(args.bs_save_folder_path):
        try:
            os.mkdir(args.bs_save_folder_path)
        except OSError:
            pass
    bs_save_folder_path_one = os.path.join(
        args.bs_save_folder_path, 'bs_' +
        os.path.basename(args.checkpoint_folder_path))
    if not os.path.exists(bs_save_folder_path_one):
        try:
            os.mkdir(bs_save_folder_path_one)
        except OSError:
            pass


    if not os.path.exists(args.fd_save_folder_path):
        try:
            os.mkdir(args.fd_save_folder_path)
        except OSError:
            pass
    fd_save_folder_path_one = os.path.join(
        args.fd_save_folder_path, 'fd_' +
        os.path.basename(args.checkpoint_folder_path))
    if not os.path.exists(fd_save_folder_path_one):
        try:
            os.mkdir(fd_save_folder_path_one)
        except OSError:
            pass


    # creat the folder to save the predicted results
    if not os.path.exists(args.ret_save_folder_path):
        try:
            os.mkdir(args.ret_save_folder_path)
        except OSError:
            pass
    ret_save_folder_path_one = os.path.join(args.ret_save_folder_path, 'ret_' + os.path.basename(args.checkpoint_folder_path))
    if not os.path.exists(ret_save_folder_path_one):
        try:
            os.mkdir(ret_save_folder_path_one)
        except OSError:
            pass
    # load the model of extractor and predictor
    if args.task_nn == 'detection':
        model_cfg_name_nn = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
        model_pretrained_path_nn = os.path.join(current_dir,'detectron2_cfg/pretrained_models/model_final_68b088.pkl')
    elif args.task_nn == 'segmentation':
        model_cfg_name_nn = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        model_pretrained_path_nn = os.path.join(current_dir, 'detectron2_cfg/pretrained_models/model_final_2d9806.pkl')
    else:
        assert False, print("Unrecognized task:", args.task_nn)

    print('task nn', args.task_nn,
          os.path.basename(model_cfg_name_nn),
          os.path.basename(model_pretrained_path_nn))
    cfg_task_nn = get_cfg()
    cfg_task_nn.merge_from_file(
        model_zoo.get_config_file(model_cfg_name_nn))
    cfg_task_nn.MODEL.WEIGHTS = model_pretrained_path_nn
    cfg_task_nn.MODEL.DEVICE = device
    task_nn = PlayerPredictor(cfg_task_nn)

    # generate image list from file
    img_list = []
    with open(args.img_test_file, 'r') as f:
        for line in f.readlines():
            img_list.append(line.strip('\n').replace('png', 'jpg'))
    img_list.sort()
    print('image_total_num:', len(img_list))

    # load the model list
    checkpoint_list = os.listdir(args.checkpoint_folder_path)
    checkpoint_list.sort(key=lambda x: float(x.split('lambda')[-1].split('_')[0]),reverse=False)
    print(checkpoint_list)
    print('num of test models:', len(checkpoint_list))
    print('the num of test models, from {0} to {1}'.format(
        args.test_model_num_min, args.test_model_num_max))

    for j in range(args.test_model_num_min, args.test_model_num_max):
        # select the pretrained model
        checkpoint_name = checkpoint_list[j]
        checkpoint_path = os.path.join(args.checkpoint_folder_path,
                                       checkpoint_name)
        print(j, checkpoint_path)

        # the path to save the compressing file
        bs_save_lambda_path = os.path.join(bs_save_folder_path_one,
                                               'lambda'+checkpoint_name.split('lambda')[-1].split('_')[0])
        if not os.path.exists(bs_save_lambda_path):
            try:
                os.mkdir(bs_save_lambda_path)
            except OSError:
                pass

        fd_save_lambda_path = os.path.join(fd_save_folder_path_one,
                                               'lambda'+checkpoint_name.split('lambda')[-1].split('_')[0])
        if not os.path.exists(fd_save_lambda_path):
            try:
                os.mkdir(fd_save_lambda_path)
            except OSError:
                pass

        # the path to save the pre results
        file_ret_save_path = os.path.join(ret_save_folder_path_one,
                                          'lambda'+checkpoint_name.split('lambda')[-1].split('_')[0])
        if not os.path.exists(file_ret_save_path):
            try:
                os.mkdir(file_ret_save_path)
            except OSError:
                pass

        # check the tested image num
        log_name = find_testimg_num(args, file_ret_save_path)

        # initialize the comppress model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net = model_feature_ar[args.model]()

        net = net.to(device)
        net.load_state_dict(checkpoint["state_dict"])

        # create the file to save the pre results
        output_fname = os.path.join(
            file_ret_save_path,checkpoint_name.split('lambda')[0][:-1] +
            '_output_{}.txt'.format(args.results_num))
        if os.path.isfile(output_fname):
            of = open(output_fname, 'a')
        else:
            of = open(output_fname, 'a')
            #write header
            if args.task_nn == 'detection':
                of.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax\n')
            else:
                of.write(
                    'ImageID,LabelName,Score,XMin,XMax,YMin,YMax,ImageWidth,ImageHeight,Mask\n'
                )

        # coco class
        coco_classes_fname = args.cococlass_path
        with open(coco_classes_fname, 'r') as f:
            coco_classes = f.read().splitlines()

        encode_decode_and_predict(args, net, task_nn,
                                  bs_save_lambda_path, fd_save_lambda_path, file_ret_save_path,
                                  img_list, of, coco_classes, log_name, device)
        of.close()