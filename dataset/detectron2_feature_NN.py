# Copyright (c) 2020-2021 Nokia Corporation and/or its subsidiary(-ies). All rights reserved.
# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

import os
import struct
import numpy as np
import torch, torchvision
import detectron2.data.transforms as T
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

from detectron2.utils.logger import setup_logger
setup_logger()

import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling.meta_arch import GeneralizedRCNN




class PlayerPredictor():
  def __init__(self, cfg):
    self.cfg = cfg.clone()  # cfg can be modified by model
    self.model = build_model(self.cfg)

    self.model.eval()
    if len(cfg.DATASETS.TEST):
      self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    checkpointer = DetectionCheckpointer(self.model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    self.aug = T.ResizeShortestEdge(
      [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    self.input_format = cfg.INPUT.FORMAT
    assert self.input_format in ["RGB", "BGR"], self.input_format

  def __call__(self, original_image, player = None, do_postprocess: bool = True, save_player: bool = False, using_saved_player: bool = False):
    ''' 
        original_image:the test image for detection or segmentation
        player: the P layer data from other programs
        save_player: whether return the extracted P layer features
        using_saved_player: whether using the input P layer data from other programs to generate the fininal predicted results
        
        This function has three purpose:
        1.when save_player==False and using_saved_player==Flase, it can generate the results of detection or sgementation from image using the pretrained model;
        2.when save_player==True and using_saved_player==Flase, it can be a feature extractor using the pretrained model and return the feature data;
        3.when save_player==False and using_saved_player==True, it can be a predictor and use the input feature data from other programs to generate the predicted results.
    '''
    
    with torch.no_grad():  
      # Apply pre-processing to image.
      if self.input_format == "RGB":
        # whether the model expects BGR inputs or RGB
        original_image = original_image[:, :, ::-1]
      height, width = original_image.shape[:2]
      image = self.aug.get_transform(original_image).apply_image(original_image)
      image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

      inputs = {"image": image, "height": height, "width": width}

      batched_inputs = [inputs]

      images = self.model.preprocess_image(batched_inputs)
      
      if not using_saved_player:
        features = self.model.backbone(images.tensor) # player generate by model    
      else:
        features = player # generate by saved file

      if save_player:
          return features

     ###########
      if self.model.proposal_generator is not None:
        proposals, _ = self.model.proposal_generator(images, features, None)
      else:
        assert "proposals" in batched_inputs[0]
        proposals = [x["proposals"].to(self.model.device) for x in batched_inputs]

      results, _ = self.model.roi_heads(images, features, proposals, None)

      if do_postprocess:
        assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
        return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
      else:
        return results