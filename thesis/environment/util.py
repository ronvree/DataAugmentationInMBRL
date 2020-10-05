
import cv2

import numpy as np

import torch


def preprocess_observation(observation: np.ndarray, bit_depth: int) -> torch.Tensor:
    """
    Perform the preprocessing steps for raw observations from the environment model
    :param observation: the observation obtained from the environment model as numpy ndarray
                        shape: (width, height, num_channels)
    :param bit_depth: the bit depth of the resulting observation
    :return: an observation tensor
                shape: (num_channels, 64, 64)
    """
    # Resize observation
    o = cv2.resize(observation, (64, 64), interpolation=cv2.INTER_LINEAR)

    # Move channel dimension, cast to tensor
    o = torch.tensor(o.transpose(2, 0, 1), dtype=torch.float32)

    # Normalize to -0.5, 0.5 range with given bit depth
    o.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)
    # Dequantize the image
    o.add_(torch.rand_like(o).div_(2 ** bit_depth))

    return o


def preprocess_observation_tensor(observation: torch.Tensor, bit_depth: int) -> torch.Tensor:
    """
    Perform the preprocessing steps for raw observations from the environment model
    :param observation: the observation obtained from the environment model as numpy ndarray
                        shape: (width, height, num_channels)
    :param bit_depth: the bit depth of the resulting observation
    :return: an observation tensor
                shape: (num_channels, 64, 64)
    """
    # Cast to float32
    o = observation.to(dtype=torch.float32)
    # Normalize to -0.5, 0.5 range with given bit depth
    o.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)
    # Dequantize the image
    o.add_(torch.rand_like(o).div_(2 ** bit_depth))

    return o


def postprocess_observation(observation: torch.Tensor,
                            bit_depth: int,
                            dtype=torch.uint8,
                            ) -> torch.Tensor:
    """
    Undo (most of) the preprocessing steps
    :param observation: the observation tensor to process
                        shape: (num_channels, 64, 64)
    :param bit_depth: the bit depth used to preprocess the original observation
    :param dtype: the datatype to which the observation should be cast (unint8 by default)
    :return: an observation tensor (values in range 0-255)
                shape: (num_channels, 64, 64)
    """
    observation = observation.cpu().detach()

    observation = torch.clamp(torch.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1)\
        .to(dtype)

    return observation
