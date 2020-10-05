
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

    raise Exception('Data augmentation function not recognized!')


if __name__ == '__main__':

    from torchvision.utils import save_image

    _img = torch.randn(10, 4, 3, 50, 50)

    _img_augmented = from_keyword('random_translate')(_img)

    save_image(_img_augmented[0, :, ...], 'test_augmentation.png')




