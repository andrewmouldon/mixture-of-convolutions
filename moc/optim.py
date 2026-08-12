import math


def moc_fan_in_std(k: int, kernel_size: int) -> float:
    return math.sqrt(k / (3.0 * kernel_size))


def match_lr_to_init_std(
    init_std: float,
    reference_std: float = 0.002,
    reference_lr: float = 1e-3,
) -> float:
    return reference_lr * init_std / reference_std


def match_weight_decay_to_lr(
    lr: float,
    reference_lr: float = 1e-3,
    reference_weight_decay: float = 0.1,
) -> float:
    return reference_weight_decay * reference_lr / lr