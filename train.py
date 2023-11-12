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

import argparse
import random
import shutil
import sys
import time
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

current_dir=os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models

from models.baseline_channel256 import JointAutoregressiveHierarchicalPriors_Channel256, Cheng2020Attention_Channel256
from dataset.dataset_feature import FeaturesFromImg_SingleLayer


# add model and dataloader
model_feature_compression = {}
model_feature_compression['JointAutoregressiveHierarchicalPriors_Channel256'] = JointAutoregressiveHierarchicalPriors_Channel256
model_feature_compression['Cheng2020Attention_Channel256'] = Cheng2020Attention_Channel256
dataloader_types = {}
dataloader_types['FeaturesFromImg_SingleLayer'] = FeaturesFromImg_SingleLayer

quality_lambda_dict = {0: 0.0009, 
                       1: 0.0018, 
                       2: 0.0035, 
                       3: 0.0067,
                       4: 0.0130, 
                       5: 0.0250, 
                       6: 0.0483, 
                       7: 0.0932,
                       8: 0.1800, 
                       9: 0.36,
                       10: 0.72,
                       11: 1.44,
                       12: 2.0}


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

def findCheckPoint(modelFolder):
    if not os.path.exists(modelFolder):
        return -1
    maxEpoch=-1
    fileList=os.listdir(modelFolder)
    modelName=''
    for fname in fileList:
        if 'epoch' in fname:
            checkpointEpoch=int(fname.split('epoch')[1].split('_')[0])
            if checkpointEpoch>maxEpoch:
                maxEpoch=checkpointEpoch
                modelName=fname
    return maxEpoch, modelName

def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, log_file
):
    model.train()
    device = next(model.parameters()).device

    avg_loss = AverageMeter()
    avg_bpp_loss = AverageMeter()
    avg_mse_loss = AverageMeter()
    avg_aux_loss = AverageMeter()

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        avg_aux_loss.update(aux_loss.item())
        avg_mse_loss.update(out_criterion["mse_loss"].item())
        avg_bpp_loss.update(out_criterion["bpp_loss"].item())
        avg_loss.update(out_criterion["loss"].item())
        

        if i % 500 == 0:
            content = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) +\
                f"Train epoch {epoch}: [" +\
                f"{i*len(d)}/{len(train_dataloader.dataset)}" +\
                f" ({100. * i / len(train_dataloader):.0f}%)]" +\
                f'\tLoss: {out_criterion["loss"].item():.3f} |' +\
                f'\tMSE loss: {out_criterion["mse_loss"].item():.5f} |' +\
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.3f} |' +\
                f"\tAux loss: {aux_loss.item():.3f}"
            print(content)
            log_file.write(content+'\n')

    content = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) +\
    f"Train epoch {epoch}: Average losses:" +\
    f"\tLoss: {avg_loss.avg:.3f} |" +\
    f"\tMSE loss: {avg_mse_loss.avg:.5f} |" +\
    f"\tBpp loss: {avg_bpp_loss.avg:.3f} |" +\
    f"\tAux loss: {avg_aux_loss.avg:.3f}\n"

    print(content)
    log_file.write(content + '\n')


def test_epoch(epoch, test_dataloader, model, criterion, log_file):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    content = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) +\
        f"Test epoch {epoch}: Average losses:" +\
        f"\tLoss: {loss.avg:.3f} |" +\
        f"\tMSE loss: {mse_loss.avg:.5f} |" +\
        f"\tBpp loss: {bpp_loss.avg:.3f} |" +\
        f"\tAux loss: {aux_loss.avg:.3f}\n"
    
    print(content)
    log_file.write(content+'\n')

    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename),"checkpoint_best_loss.pth.tar"))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="training scale-adaptive entropy model.")

    parser.add_argument(
        "--task_extractor",
        default="opimg_det",
        help="Task",
    )

    parser.add_argument(
        "-m",
        "--model",
        default="JointAutoregressiveHierarchicalPriors_Channel256",
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quality-level",
        type=int,
        default=10,
        help="Quality level (default: %(default)s)",
    )
    parser.add_argument(
        "--dataloader_type",
        type=str,
        default='FeaturesFromImg_SingleLayer',
        help="dataloader type",
    )
    parser.add_argument(
        "-d", "--dataset", default='/home/dingding/workspace/MM/Dataset/FCM', type=str, help="Training dataset"
    )
    parser.add_argument(
        "--train_dataset", default="train50k", type=str, help="Training dataset"
    )
    parser.add_argument(
        "--test_dataset", default="opimg_det_test", type=str, help="Testing dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=300,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=2,
        help="Dataloaders threads (default: %(default)s)",
    )
    # parser.add_argument(
    #     "--lambda",
    #     dest="lmbda",
    #     type=float,
    #     default=1e-2,
    #     help="Bit-rate distortion parameter (default: %(default)s)",
    # )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=8,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", default=17, type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")

    parser.add_argument("--save_location", default = os.path.join(current_dir,'saveModels'), type=str, help="Path to save ckpt")
    
    parser.add_argument("--model_pretrained_path", default = os.path.join(current_dir,'detectron2_cfg/pretrained_models'), type=str, help="Path to save ckpt")


    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    ng = torch.cuda.device_count()
    print("Cuda Devices:%d" %ng)
    for i in range(ng):
        print(torch.cuda.get_device_properties(i))

    args.lmbda=quality_lambda_dict[int(args.quality_level)]

    # create the save dir
    if args.save:
        if not os.path.exists(args.save_location):
            os.mkdir(args.save_location)
        save_location=os.path.join(args.save_location,args.task_extractor,args.model,str(args.lmbda))
        if not os.path.exists(save_location):
            os.makedirs(save_location)

        log_file = open(os.path.join(save_location, 'train_log.txt'), 'w')
    else:
        log_file = open('./train_log.txt', 'w')

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    else:
        torch.manual_seed(17)
        random.seed(17)  

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(size=args.patch_size,pad_if_needed=True,fill=0,padding_mode='constant')])
    
    test_transforms = transforms.Compose(
        [transforms.RandomCrop(size=args.patch_size,pad_if_needed=True,fill=0,padding_mode='constant')])

    #assert args.task_extractor+'_test' == args.test_dataset

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # initialize the dataloader of features
    train_dataset =dataloader_types[args.dataloader_type](args.dataset, transform=train_transforms, split=args.train_dataset,  data_split=5, patch_size=args.patch_size,task_extractor=args.task_extractor,model_pretrained_path=args.model_pretrained_path,device=device)
    test_dataset = dataloader_types[args.dataloader_type](args.dataset, transform=test_transforms, split=args.test_dataset, data_split=1, patch_size=args.patch_size,task_extractor=args.task_extractor,model_pretrained_path=args.model_pretrained_path,device=device)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = model_feature_compression[args.model]()
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.save_location:
        if os.path.exists(args.save_location):
            save_location=os.path.join(args.save_location,args.task_extractor,args.model, str(args.lmbda))
            if os.path.exists(save_location):
                last_epoch_temp, ckptName =  findCheckPoint(save_location)
                print('%-30s%-20s' %('find latest model epoch',last_epoch_temp))
                if not ckptName =='':
                    args.checkpoint=os.path.join(save_location,ckptName)
                    last_epoch = last_epoch_temp

    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])


    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print('%-30s%-20s' %('Learning rate',optimizer.param_groups[0]['lr']))
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            log_file
        )

        if epoch%5==0 or epoch>=int(args.epochs-10):

            loss = test_epoch(epoch, test_dataloader, net, criterion, log_file)
            lr_scheduler.step(loss)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if args.save:
                model_path=os.path.join(save_location,args.model+f'_lambda{args.lmbda}_epoch{epoch}_checkpoint.pth.tar')
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best, 
                    filename=model_path
                )

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main(sys.argv[1:])