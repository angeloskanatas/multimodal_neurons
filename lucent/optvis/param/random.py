# Copyright 2024 Multimodal Neurons. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import, division, print_function

import torch
import numpy as np
from torch.fft import rfftn, irfftn

from lucent.optvis.param.spatial import rfft2d_freqs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TORCH_VERSION = torch.__version__


# Decorrelation
color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")

max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt

color_mean = [0.48, 0.46, 0.41]


def _linear_decorrelate_color(tensor):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t_permute = tensor.permute(0, 2, 3, 1)
    t_permute = torch.matmul(t_permute, torch.tensor(color_correlation_normalized.T).to(device))
    tensor = t_permute.permute(0, 3, 1, 2)
    return tensor

    
def image_sample(shape, decorrelate=True, sd=None, decay_power=1):
    _, raw_spatial_func = rand_fft_image(shape, sd=sd, decay_power=decay_power)
    raw_spatial = raw_spatial_func()
    if decorrelate:
        output = _linear_decorrelate_color(raw_spatial)
    else:
        output = raw_spatial
    return torch.sigmoid(output)

    
def rand_fft_image(shape, sd=None, decay_power=1):
    batch, channels, h, w = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (batch, channels) + freqs.shape + (2,) # 2 for imaginary and real components
    sd = sd or 0.01

    spectrum_real_imag_t = (torch.randn(*init_val_size) * sd).to(device)

    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale = torch.tensor(scale).float()[None, None, ..., None].to(device)

    def inner():
        scaled_spectrum_t = scale * spectrum_real_imag_t
        if TORCH_VERSION >= "1.7.0":
            import torch.fft
            if type(scaled_spectrum_t) is not torch.complex64:
                scaled_spectrum_t = torch.view_as_complex(scaled_spectrum_t)
            image = torch.fft.irfftn(scaled_spectrum_t, s=(h, w), norm='ortho')
        else:
            import torch
            image = torch.irfft(scaled_spectrum_t, 2, normalized=True, signal_sizes=(h, w))
        image = image[:batch, :channels, :h, :w]
        magic = 4.0 # Magic constant from Lucid library; increasing this seems to reduce saturation
        image = image / magic
        return image
    return [spectrum_real_imag_t], inner
