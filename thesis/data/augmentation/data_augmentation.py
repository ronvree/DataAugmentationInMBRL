import random

import numpy as np

import torch


"""
    Data augmentation functions based on the RAD module
"""


def random_translate(imgs: torch.Tensor, out: int = 78) -> torch.Tensor:
    """

    :param imgs:
    :param out:
    :return:
    """
    episode_length, batch_size, c, h, w = imgs.shape
    w1 = np.random.randint(0, out - w + 1, batch_size)
    h1 = np.random.randint(0, out - h + 1, batch_size)

    target = torch.zeros((episode_length, batch_size, c, out, out), dtype=imgs.dtype, device=imgs.device)

    for t, img_batch in enumerate(imgs):
        for i, (img, w11, h11) in enumerate(zip(img_batch, w1, h1)):
            target[t, i, :, w11:w11+w, h11:h11+h] = img

    return target


def fixed_translate(imgs: torch.Tensor, out: int = 78) -> torch.Tensor:
    """

    :param imgs:
    :param out:
    :return:
    """
    episode_length, batch_size, c, h, w = imgs.shape

    target = torch.zeros((episode_length, batch_size, c, out, out), dtype=imgs.dtype, device=imgs.device)
    target[:, :, :, 0:h, 0:w] = imgs

    return target


def vertical_flip(imgs: torch.Tensor) -> torch.Tensor:
    imgs = torch.flip(imgs, dims=[4])
    return imgs


def horizontal_flip(imgs: torch.Tensor) -> torch.Tensor:
    imgs = torch.flip(imgs, dims=[3])
    return imgs


def negate(t: torch.Tensor) -> torch.Tensor:
    return -t


"""

    STATE-ACTION AUGMENTATIONS

"""


def maybe_vertical_flip_and_negate(imgs: torch.Tensor,
                                   actions: torch.Tensor,
                                   imgs_: torch.Tensor,
                                   actions_: torch.Tensor) -> tuple:
    if random.random() > 0.5:
        imgs = vertical_flip(imgs)
        actions = negate(actions)
        imgs_ = vertical_flip(imgs_)
        actions_ = negate(actions_)

        return imgs, actions, imgs_, actions_
    else:
        return imgs, actions, imgs_, actions_


"""

    OTHER
    
"""


def from_keyword(kw: str) -> callable:
    """
    Get the data augmentation function that matches the given keyword
    :param kw: a keyword indicating some data augmentation
    :return: the data augmentation function.
    """
    if kw == 'random_translate':
        return random_translate
    if kw == 'fixed_translate':
        return fixed_translate
    if kw == 'vertical_flip':
        return vertical_flip
    if kw == 'horizontal_flip':
        return horizontal_flip
    if kw == 'negate':
        return negate
    if kw == 'maybe_vertical_flip_and_negate':
        return maybe_vertical_flip_and_negate

    raise Exception('Data augmentation function not recognized!')


if __name__ == '__main__':

    from torchvision.utils import save_image

    # _img = torch.randn(10, 4, 3, 50, 50)
    _img = torch.randn(10, 4, 3, 50, 50) * torch.eye(50)

    _img_augmented = from_keyword('vertical_flip')(_img)

    save_image(_img[0, :, ...], 'test_image.png')
    save_image(_img_augmented[0, :, ...], 'test_augmentation.png')




