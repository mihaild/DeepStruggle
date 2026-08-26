"""Twilight Struggle Neural Network Architectures."""
from .coldwar_net import ColdWarNet, create_coldwar_net
from .coldwar_net_v2 import ColdWarNetV2, create_coldwar_net_v2
from .coldwar_net_v3 import ColdWarNetV3, create_coldwar_net_v3

__all__ = [
    "ColdWarNet",
    "create_coldwar_net",
    "ColdWarNetV2",
    "create_coldwar_net_v2",
    "ColdWarNetV3",
    "create_coldwar_net_v3",
]
