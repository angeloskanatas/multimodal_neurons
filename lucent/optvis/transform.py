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
import torch.nn.functional as F
from torchvision.transforms import Normalize
import numpy as np
import kornia
from kornia.geometry.transform import translate

from lucent.optvis import param


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
KORNIA_VERSION = kornia.__version__


def jitter(d):
    assert d > 1, "Jitter parameter d must be more than 1, currently {}".format(d)

    def inner(image_t):
        dx = np.random.choice(d)
        dy = np.random.choice(d)
        return translate(image_t, torch.tensor([[dx, dy]]).float().to(device))

    return inner


def pad(w, mode="reflect", constant_value=0.5):
    if mode != "constant":
        constant_value = 0

    def inner(image_t):
        return F.pad(image_t, [w] * 4, mode=mode, value=constant_value,)

    return inner


def random_scale(scales):
    def inner(image_t):
        scale = np.random.choice(scales)
        shp = image_t.shape[2:]
        scale_shape = [_roundup(scale * d) for d in shp]
        scale_shape = tuple(int(dim) for dim in scale_shape)
        pad_x = max(0, _roundup((shp[1] - scale_shape[1]) / 2))
        pad_y = max(0, _roundup((shp[0] - scale_shape[0]) / 2))
        
        upsample = torch.nn.Upsample(
            size=scale_shape, mode="bilinear", align_corners=True
        )
        return F.pad(upsample(image_t), [pad_y, pad_x] * 2)

    return inner


def random_rotate(angles, units="degrees"):
    def inner(image_t):
        b, _, h, w = image_t.shape
        # kornia takes degrees
        alpha = _rads2angle(np.random.choice(angles), units)
        angle = torch.ones(b) * alpha
        if KORNIA_VERSION < '0.4.0':
            scale = torch.ones(b)
        else:
            scale = torch.ones(b, 2)
        center = torch.ones(b, 2)
        center[..., 0] = (image_t.shape[3] - 1) / 2
        center[..., 1] = (image_t.shape[2] - 1) / 2
        M = kornia.get_rotation_matrix2d(center, angle, scale).to(device)
        rotated_image = kornia.warp_affine(image_t.float(), M, dsize=(h, w))
        return rotated_image

    return inner
    
    
def collapse_alpha_random(sd=0.5):
    def inner(image_t):
        rgb, a = image_t[:, :3, :, :], image_t[:, 3:4, :, :]
        rgb_shape = list(rgb.size())
        # rand_img = torch.rand(*rgb_shape, device=t_image.device) # -> this needs further investigation
        # rand_img = rand_img * sd + (1 - sd) * 0.5
        rand_img = param.random.image_sample(rgb_shape, sd=sd) # lucid implementation
        output = a * rgb + (1 - a) * rand_img
        return output

    return inner


def compose(transforms):
    def inner(x):
        for transform in transforms:
            x = transform(x)
        return x

    return inner


def _roundup(value):
    return np.ceil(value).astype(int)


def _rads2angle(angle, units):
    if units.lower() == "degrees":
        return angle
    if units.lower() in ["radians", "rads", "rad"]:
        angle = angle * 180.0 / np.pi
    return angle


def normalize():
    # ImageNet normalization for torchvision models
    # see https://pytorch.org/docs/stable/torchvision/models.html
    normal = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def inner(image_t):
        return torch.stack([normal(t) for t in image_t])

    return inner
    
def crop_or_pad_to(height, width):
    """Ensures the specified spatial shape by either padding or cropping.
    Meant to be used as a last transform for architectures insisting on a specific
    spatial shape of their inputs.
    """
    def inner(t_image):
        c, h, w = t_image.shape[1:]
        pad_vert = max(height - h, 0)
        pad_horz = max(width - w, 0)
        pad_top = pad_vert // 2
        pad_bottom = pad_vert - pad_top
        pad_left = pad_horz // 2
        pad_right = pad_horz - pad_left
        if pad_vert > 0 or pad_horz > 0:
            t_image = F.pad(t_image, (pad_left, pad_right, pad_top, pad_bottom))
        if h > height or w > width:
            top = (h - height) // 2
            left = (w - width) // 2
            t_image = t_image[:, :, top:top + height, left:left + width]
        return t_image

    return inner


def preprocess_inceptionv1():
    # Original Tensorflow's InceptionV1 model
    # takes in [-117, 138]
    # See https://github.com/tensorflow/lucid/blob/master/lucid/modelzoo/other_models/InceptionV1.py#L56
    # Thanks to ProGamerGov for this!
    return lambda x: x * 255 - 117


standard_transforms = [
    pad(12, mode="constant", constant_value=0.5),
    jitter(8),
    random_scale([1 + (i - 5) / 50.0 for i in range(11)]),
    random_rotate(list(range(-10, 11)) + 5 * [0]),
    jitter(4),
]
