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
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '4'
current_dir=os.path.dirname(os.path.abspath(__file__))


import torch
import torch.nn as nn
from workspace.MM.FCM.PyramidContext_Entropy_Model.models.cross_scale_model import JointAutoregressiveHierarchicalPriors_Channel256


if __name__ == "__main__":

    ## 256 channels nomerge

    # modified the epoch of the updated models
    epoch = 100
    
    task='opimg_det'
    
    orgModelPathes = [
        "JointAutoregressiveHierarchicalPriors_Channel256_lambda0.0932_epoch{}_checkpoint.pth.tar".format(epoch),
    ]
    newModelPathes = [
        "JointAutoregressiveHierarchicalPriors_Channel256_lambda0.0932_new_checkpoint.pth.tar",
    ]
    lambda_list = [
        '0.0932',
    ]

    # midify the saving path for the updated models
    ModelSavePath = os.path.join(current_dir,'saveModels')
    for idx in range(len(orgModelPathes)):
        OrgModelSaveFolder=os.path.join(ModelSavePath,task)
        NewModelSaveFolder = os.path.join(
        ModelSavePath, 'opimg_'+task.split('_')[-1]+'_pretrained_model_{}'.format(epoch))
        if not os.path.exists(NewModelSaveFolder):
            os.mkdir(NewModelSaveFolder)
            
        OrgModelSaveFolder = os.path.join(OrgModelSaveFolder, lambda_list[idx])
        # NewModelSaveFolder = os.path.join(NewModelSaveFolder, lambda_list[idx])

        if not os.path.exists(NewModelSaveFolder):
          os.mkdir(NewModelSaveFolder)

        print(idx, OrgModelSaveFolder)
        print(idx, NewModelSaveFolder)

        orgModelPath = os.path.join(OrgModelSaveFolder, orgModelPathes[idx])
        newModelPath = os.path.join(NewModelSaveFolder, newModelPathes[idx])
        net = JointAutoregressiveHierarchicalPriors_Channel256()
        device = "cuda"
        net = net.to(device)
        checkpoint = torch.load(orgModelPath, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        net.update()

        torch.save({
            "epoch": epoch,
            "state_dict": net.state_dict(),
        }, newModelPath)
