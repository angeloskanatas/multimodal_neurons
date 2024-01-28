# Copyright 2020 The Lucent Authors. All Rights Reserved.
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


def to_valid_rgb(image_f, decorrelate=False, alpha=False):
    def inner():
        image_tensor = image_f()
        image_tensor_rgb = image_tensor[:, :3, :, :]
        if decorrelate:
            image_tensor_rgb = _linear_decorrelate_color(image_tensor_rgb)
        image = torch.sigmoid(image_tensor_rgb)
        if alpha:
            image_tensor_alpha = image_tensor[:, 3:, :, :]
            alpha_channel = torch.sigmoid(image_tensor_alpha)
            image = torch.cat((image, alpha_channel), dim=1)
        return image
    return inner
