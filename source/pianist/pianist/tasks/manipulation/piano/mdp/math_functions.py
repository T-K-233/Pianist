import torch


def gaussian(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Returns a gaussian function of the input tensor.

    Args:
        x: The input tensor.
        mean: The mean of the gaussian.
        std: The standard deviation of the gaussian.
    """
    assert std > 0, "Standard deviation must be positive."
    return torch.exp(-0.5 * torch.square(x - mean) / (std**2))


def windowed_gaussian(
    x: torch.Tensor,
    lower: float,
    upper: float,
    std: float,
) -> torch.Tensor:
    """
    Returns 1 when `x` falls inside the bounds, gaussianly decreasing outside the bounds.

    This function implements the following formula:
        y = {
            x < L: e^((L-x)^2)/(sigma^2),
            x > U: e^((x-U)^2)/(sigma^2),
            L < x < U: 1
        }

    Args:
        x: The input tensor.
        lower: The lower bound.
        upper: The upper bound.
        std: The standard deviation of the gaussian.
    """
    assert lower < upper, "Lower bound must be less than upper bound."
    assert std > 0, "Standard deviation must be positive."

    x = torch.where(
        x < lower,
        gaussian(x, lower, std),
        torch.where(
            x > upper,
            gaussian(x, upper, std),
            1.0,
        ),
    )
    return x
