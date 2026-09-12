"""Neural network model architectures for Twilight Struggle AI."""

from .coldwar_net import ColdWarNet, create_coldwar_net
from .coldwar_net_v2 import ColdWarNetV2, create_coldwar_net_v2

__all__ = [
    "ColdWarNet",
    "create_coldwar_net",
    "ColdWarNetV2",
    "create_coldwar_net_v2",
]
