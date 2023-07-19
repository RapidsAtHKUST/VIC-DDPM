from models.guided_ddpm_unet import UNetModel
from utils.galaxy_data_utils.transform_util import *
import torch
import numpy as np

def posenc(x, L_embed=4):
  rets = [x]
  for i in range(0, L_embed):
    for fn in [torch.sin, torch.cos]:
      rets.append(fn(2.*3.14159265*(i+1) * x))
  return torch.cat(rets, dim=-1)
  
def calcB(m=1024, d=2, sigma=1.0):
    B = torch.randn(m, d)*sigma
    return B.cuda()
    
def fourierfeat_enc(x, B):
    feat = torch.cat([#torch.sum(x**2, -1, keepdims=True), ## new
                      x, ## new
                      torch.cos(2*3.14159265*(x @ B.T)),
                      torch.sin(2*3.14159265*(x @ B.T))], -1)
    return feat

class PE_Module(torch.nn.Module):
    def __init__(self, type, embed_L):
        super(PE_Module, self).__init__()

        self.embed_L= embed_L
        self.type=type

    def forward(self, x):
        if self.type == 'posenc':
            return posenc(x, L_embed=self.embed_L)

        elif self.type== 'fourier':
            return fourierfeat_enc(x, B=self.embed_L)

class VicUnetModel(UNetModel):


    def __init__(self, image_size, in_channels, *args, **kwargs):
        assert in_channels == 2, "mri image is considered"
        # we use in_channels * 2 because image_dir is also input.
        super().__init__(image_size, in_channels * 2, *args, **kwargs)
        self.uv_dense = np.load(".data/uv_dense.npy")
        self.uv_dense = torch.tensor(self.uv_dense)
        self.B = torch.nn.Parameter(calcB(m=10, d=2, sigma=5.0), requires_grad=False)
        self.pe_encoder = PE_Module(type='fourier', embed_L= self.B)


    def forward(self, x, timesteps, uv_coords, image_dir, vis_sparse):
        """

        :param x: the [N x 2 x H x W] tensor of inputs, x_t at time t.
        :param timesteps: a batch of timestep indices.
        :param image_dir: the [N x 2 x H x W] tensor, dirty img.
        :param visibility: sparse visibility before encoding
        :return: noise estimation
        """
        pos = uv_coords

        pe_uv = self.pe_encoder(uv_coords)

        visibility = torch.cat([pe_uv, vis_sparse], dim=-1) # 32, 1660, 24
        x = th.cat([x, image_dir], dim = 1) 

        output = super().forward(x, timesteps, visibility)
        
        return output