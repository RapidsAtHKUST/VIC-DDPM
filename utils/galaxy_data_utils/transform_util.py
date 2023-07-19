"""
Based on the source code from Wu, Benjamin, et al. "Neural Interferometry: Image Reconstruction from Astronomical Interferometers using Transformer-Conditioned Neural Fields." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 3. 2022.
"""

import numpy as np
import torch as th
from utils.dataset_utils.data_ehtim_cont import *


# -------- FFT transform --------

def to_img_th(s_vis_real, s_vis_imag, uv_dense):
    nF = 128
    s_vis_imag[:, 0, 0] = 0
    s_vis_imag[:, 0, nF//2] = 0
    s_vis_imag[:, nF//2, 0] = 0
    s_vis_imag[:, nF//2, nF//2] = 0

    s_fft = s_vis_real + 1j * s_vis_imag

    # NEW: set border to zero to counteract weird border issues
    s_fft[:, 0, :] = 0.0
    s_fft[:, :, 0] = 0.0
    s_fft[:, :, -1] = 0.0
    s_fft[:, -1, :] = 0.0

    eht_fov = 1.4108078120287498e-09
    max_base = 8368481300.0
    img_res = 128
    scale_ux = max_base * eht_fov / img_res
    b = s_vis_imag.shape[0]
    uv_dense_per = uv_dense.unsqueeze(0).repeat(s_vis_real.size(0), 1, 1)
    u_dense, v_dense = uv_dense_per[:, :, 0].unique(dim=1), uv_dense_per[:, :, 1].unique(dim=1)
    u_dense = torch.linspace(u_dense.min(), u_dense.max(), len(u_dense[0]) // 2 * 2).unsqueeze(0).repeat(s_vis_real.size(0), 1).to(u_dense)
    v_dense = torch.linspace(v_dense.min(), v_dense.max(), len(v_dense[0]) // 2 * 2).unsqueeze(0).repeat(s_vis_real.size(0), 1).to(u_dense)
    uv_arr = torch.cat([u_dense.unsqueeze(-1), v_dense.unsqueeze(-1)], dim=-1)
    uv_arr = ((uv_arr + 0.5) * 2 - 1.) * scale_ux

    img_recon = make_im_torch(uv_arr[0], s_fft, img_res, eht_fov, norm_fact=1., return_im=True)

    img_real = img_recon.real.squeeze(0)
    img_imag = img_recon.imag.squeeze(0)
    img_recon = torch.cat([img_real, img_imag], dim=0)
    img_recon = img_recon.reshape(b, 2, 128, 128)

    return img_recon.float()





# -------- dtype transform --------

def complex2real_np(x):
    """
    Change a complex numpy.array to a real array with two channels.

    :param x: numpy.array of complex with shape of (h, w).

    :return: numpy.array of real with shape of (2, h, w).
    """
    return np.stack([x.real, x.imag])


def real2complex_np(x):
    """
    Change a real numpy.array with two channels to a complex array.

    :param x: numpy.array of real with shape of (2, h, w).

    :return: numpy.array of complex64 with shape of (h, w).
    """
    complex_x = np.zeros_like(x[0, ...], dtype=np.complex64)
    complex_x.real, complex_x.imag = x[0], x[1]
    return complex_x


def np2th(x):
    return th.tensor(x)


def th2np(x):
    return x.detach().cpu().numpy()


def np_comlex_to_th_real2c(x):
    """
    Transform numpy.array of complex to th.Tensor of real with 2 channels.

    :param x: numpy.array of complex with shape of (h, w).

    :return: th.Tensor of real with 2 channels with shape of (h, w, 2).
    """
    return np2th(complex2real_np(x).transpose((1, 2, 0)))


def th_real2c_to_np_complex(x):
    """
    Transform th.Tensor of real with 2 channels to numpy.array of complex.

    :param x: th.Tensor of real with 2 channels with shape of (h, w, 2).

    :return: numpy.array of complex with shape of (h, w).
    """
    return real2complex_np(th2np(x.permute(2, 0, 1)))


def th2np_magnitude(x):
    """
    Compute the magnitude of torch.Tensor with shape of (b, 2, h, w).

    :param x: th.Tensor of real with 2 channels with shape of (b, 2, h, w).

    :return: numpy.array of real with shape of (b, h, w).
    """
    x = th2np(x)
    return np.sqrt(x[:, 0, ...] ** 2 + x[:, 1, ...] ** 2)
