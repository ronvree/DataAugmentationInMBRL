
import torch


def batch_tensors(*ts) -> torch.Tensor:
    """
    Convert multiple tensors of the same shape to a single batch tensor
    The batch size is the length of xs
    :param ts: the tensors that should be batched
    :return: a single batch tensor containing the individual tensors
    """
    return torch.cat([t.unsqueeze(0) for t in ts], dim=0)


def sample_normal(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Sample from a normal distribution while preserving the gradient. torch.normal does not do this
    :param mean: a mean tensor parameterizing the normal distribution
    :param std: a standard deviation tensor parameterizing the normal distribution
    :return: a tensor obtained from sampling the distribution
    """
    return mean + std * torch.randn_like(std)
