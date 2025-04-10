from typing import Any
from collections import defaultdict

from omegaconf import DictConfig, OmegaConf, ListConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode


def get_allowed_classes():
    return [DictConfig, ListConfig, OmegaConf, ContainerMetadata, Any, list, dict,
            defaultdict, int, float, AnyNode, Metadata]
