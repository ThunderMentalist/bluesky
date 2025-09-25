"""Hierarchy construction utilities for the Precision MMM package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class Hierarchy:
    """Represents the mapping between tacticals, platforms, and channels."""

    channel_names: List[str]
    platform_names: List[str]
    tactical_names: List[str]
    M_tp: np.ndarray  # tacticals -> platforms
    M_tc: np.ndarray  # tacticals -> channels
    t_to_p: np.ndarray  # tactical index -> platform index
    p_to_c: np.ndarray  # platform index -> channel index

    @property
    def num_channels(self) -> int:
        return len(self.channel_names)

    @property
    def num_platforms(self) -> int:
        return len(self.platform_names)

    @property
    def num_tacticals(self) -> int:
        return len(self.tactical_names)


def _ensure_unique(values: Sequence[str], level: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {level} names detected: {values}")


def build_hierarchy(
    spec: Dict[str, Dict[str, Sequence[str]]],
    *,
    keep_order: bool = True,
) -> Hierarchy:
    """Create hierarchy mappings from a nested specification.

    Args:
        spec: A nested mapping of channel -> platform -> list of tacticals.
        keep_order: If ``True`` (default), preserve the insertion order of the
            specification. If ``False``, channel, platform, and tactical names
            are sorted alphabetically.

    Returns:
        Hierarchy dataclass containing name lists and aggregation matrices.
    """

    if not spec:
        raise ValueError("spec must contain at least one channel")

    channel_names = list(spec.keys()) if keep_order else sorted(spec.keys())
    _ensure_unique(channel_names, "channel")

    platform_names: List[str] = []
    tactical_names: List[str] = []
    p_to_c_idx: List[int] = []
    t_to_p_idx: List[int] = []

    for c_idx, c_name in enumerate(channel_names):
        platforms = spec[c_name]
        if not platforms:
            raise ValueError(f"Channel '{c_name}' must contain at least one platform")
        platform_keys = list(platforms.keys()) if keep_order else sorted(platforms.keys())
        _ensure_unique(platform_keys, "platform")
        for p_name in platform_keys:
            p_idx = len(platform_names)
            platform_names.append(p_name)
            p_to_c_idx.append(c_idx)

            tacticals = platforms[p_name]
            if not tacticals:
                raise ValueError(
                    f"Platform '{p_name}' within channel '{c_name}' must contain tacticals"
                )
            tactical_list = list(tacticals) if keep_order else sorted(tacticals)
            _ensure_unique(tactical_list, "tactical")
            for tactical_name in tactical_list:
                tactical_names.append(tactical_name)
                t_to_p_idx.append(p_idx)

    num_tacticals = len(tactical_names)
    num_platforms = len(platform_names)
    num_channels = len(channel_names)

    t_to_p = np.array(t_to_p_idx, dtype=int)
    p_to_c = np.array(p_to_c_idx, dtype=int)

    M_tp = np.zeros((num_tacticals, num_platforms), dtype=np.float64)
    for tactical_idx, platform_idx in enumerate(t_to_p):
        M_tp[tactical_idx, platform_idx] = 1.0

    M_pc = np.zeros((num_platforms, num_channels), dtype=np.float64)
    for platform_idx, channel_idx in enumerate(p_to_c):
        M_pc[platform_idx, channel_idx] = 1.0

    M_tc = M_tp @ M_pc

    return Hierarchy(
        channel_names=channel_names,
        platform_names=platform_names,
        tactical_names=tactical_names,
        M_tp=M_tp,
        M_tc=M_tc,
        t_to_p=t_to_p,
        p_to_c=p_to_c,
    )
