import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
from torchsummary import summary
from torchstat import stat

from thop import profile
from thop import clever_format

from model import Cheng2020Attention_features_input256
from compressai.models.dkic import DKIC_features_input256



class Cheng2020Attention_complexity_stat(Cheng2020Attention_features_input256):
    def __init__(self, N=192, in_ch=256, out_ch=256, **kwargs):
        super(Cheng2020Attention_complexity_stat,self).__init__(N=192, in_ch=256, out_ch=256, **kwargs)

    def forward(self, x):
        bitstream=self.compress(x)
        out=self.decompress(bitstream["strings"],bitstream["shape"])
        return out
    


class DKIC_complexity_stat(DKIC_features_input256):
    def __init__(self, N=192, M=320, in_ch=256, out_ch=256, **kwargs):
        super(DKIC_complexity_stat,self).__init__(N=192, M=320, in_ch=256, out_ch=256, **kwargs)

    def forward(self, x):
        bitstream=self.compress(x)
        out=self.decompress(bitstream["strings"],bitstream["shape"])
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cheng2020attention = Cheng2020Attention_complexity_stat(N=192,
                                                          in_ch=256,
                                                          out_ch=256).to(device)
cheng2020attention.update()
dkic = DKIC_complexity_stat(N=128, M=320, in_ch=256, out_ch=256).to(device)
dkic.update()

import torch
from ptflops import get_model_complexity_info

stat_dict=dict()

C=256
H=128
W=128

cheng2020attention.eval()
macs, params = get_model_complexity_info(cheng2020attention, (C,H,W), as_strings=True, print_per_layer_stat=True, verbose=True)
macs = float(macs[:5]) * 1024 * 1024
macs= str(float(macs) / (H*W))+' K'
stat_dict['cheng2020attention']={'MACs':macs,'Parameters':params}


dkic.eval()
macs, params = get_model_complexity_info(dkic, (C,H,W), as_strings=True, print_per_layer_stat=True, verbose=True)
macs = float(macs[:5]) * 1024 * 1024
macs= str(float(macs) / (H*W))+' K'
stat_dict['dkic']={'MACs':macs,'Parameters':params}

width = 20
print('{0:^{width}}|{1:^{width}}|{2:^{width}}'.format('Model',
                                                      'MACs(K)',
                                                      'Params(M)',
                                                      width=width))

for key, value in stat_dict.items():
    print('{0:^{width}}|{1:^{width}}|{2:^{width}}'.format(key,
                                                        value['MACs'],
                                                        value['Parameters'],
                                                        width=width))
