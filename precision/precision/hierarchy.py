"""Hierarchy construction utilities for the Precision MMM package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class Hierarchy:
    """Representation of an arbitrary N-level hierarchy."""

    levels: List[str]
    names: Dict[str, List[str]]
    maps_adjacent: Dict[Tuple[str, str], np.ndarray]
    index_adjacent: Dict[Tuple[str, str], np.ndarray]

    def size(self, level: str) -> int:
        """Return the number of nodes at ``level``."""

        try:
            return len(self.names[level])
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise KeyError(f"Unknown level {level!r}") from exc

    def map(self, child: str, parent: str) -> np.ndarray:
        """Return the cumulative mapping matrix from ``child`` to ``parent`` levels."""

        child_idx = self.levels.index(child)
        parent_idx = self.levels.index(parent)
        if child_idx == parent_idx:
            size = self.size(child)
            return np.eye(size, dtype=float)
        if child_idx > parent_idx:
            raise ValueError(
                "Mapping is only defined from a lower level to a higher level in the hierarchy"
            )

        result = None
        for level_idx in range(child_idx, parent_idx):
            c = self.levels[level_idx]
            p = self.levels[level_idx + 1]
            M = self.maps_adjacent[(c, p)]
            result = M if result is None else result @ M
        assert result is not None  # for mypy; parent_idx > child_idx ensured a matrix was set
        return result

    def index_map(self, child: str, parent: str) -> np.ndarray:
        """Return the parent index for each ``child`` element."""

        child_idx = self.levels.index(child)
        parent_idx = self.levels.index(parent)
        if child_idx == parent_idx:
            return np.arange(self.size(child), dtype=int)
        if child_idx > parent_idx:
            raise ValueError(
                "Index mapping is only defined from a lower level to a higher level in the hierarchy"
            )

        indices = self.index_adjacent[(self.levels[child_idx], self.levels[child_idx + 1])].copy()
        for level_idx in range(child_idx + 1, parent_idx):
            lvl = self.levels[level_idx]
            nxt = self.levels[level_idx + 1]
            indices = self.index_adjacent[(lvl, nxt)][indices]
        return indices

    @property
    def num_channels(self) -> int:
        if "channel_names" not in self.__dict__:
            raise AttributeError("channel_names not available for this hierarchy")
        return len(self.channel_names)

    @property
    def num_platforms(self) -> int:
        if "platform_names" not in self.__dict__:
            raise AttributeError("platform_names not available for this hierarchy")
        return len(self.platform_names)

    @property
    def num_tacticals(self) -> int:
        if "tactical_names" not in self.__dict__:
            raise AttributeError("tactical_names not available for this hierarchy")
        return len(self.tactical_names)


def _compat_three_level_fields(hierarchy: Hierarchy) -> Dict[str, object]:
    """Return legacy 3-level attributes for backwards compatibility."""

    if len(hierarchy.levels) < 3:
        return {}

    bottom, middle, top = hierarchy.levels[:3]
    compat: Dict[str, object] = {
        f"{level}_names": hierarchy.names[level]
        for level in hierarchy.levels
    }

    compat.update(
        {
            "M_tp": hierarchy.map(bottom, middle),
            "M_tc": hierarchy.map(bottom, top),
            "t_to_p": hierarchy.index_map(bottom, middle),
            "p_to_c": hierarchy.index_map(middle, top),
        }
    )

    return compat


def pad_ragged_tree(
    tree_top_to_bottom: Dict[str, Any],
    levels: Sequence[str],
    *,
    placeholder_prefix: str = "__auto__",
) -> Dict[str, Any]:
    """Normalize ragged ``tree_top_to_bottom`` by inserting placeholder nodes.

    ``tree_top_to_bottom`` is expected to map top-level node names to nested
    subtrees as required by :func:`build_hierarchy`.  When some branches skip
    intermediate levels, placeholder nodes are inserted so that every path
    traverses the same ``levels`` depth.
    """

    if not isinstance(tree_top_to_bottom, dict):
        raise TypeError("tree_top_to_bottom must be a dict of top-level nodes")

    levels = list(levels)
    if not levels:
        raise ValueError("levels must be non-empty")
    top_idx = len(levels) - 1

    def ensure(level_idx: int, subtree: Any, parent_path: List[str]):
        if level_idx == 0:
            if isinstance(subtree, list):
                return subtree
            raise TypeError(f"Expected list at leaf level {levels[0]!r}")

        if level_idx == 1 and isinstance(subtree, list):
            return subtree

        if isinstance(subtree, dict):
            out: Dict[str, Any] = {}
            for child_name, child_subtree in subtree.items():
                if not isinstance(child_name, str):
                    raise TypeError("Hierarchy node names must be strings")
                out[child_name] = ensure(
                    level_idx - 1, child_subtree, parent_path + [child_name]
                )
            return out

        if isinstance(subtree, list):
            placeholder_level = levels[level_idx - 1]
            placeholder_name = (
                f"{placeholder_prefix}:{placeholder_level}:"
                f"{'/'.join(parent_path) or 'root'}"
            )
            return {
                placeholder_name: ensure(
                    level_idx - 1, subtree, parent_path + [placeholder_name]
                )
            }

        raise TypeError(
            f"Invalid subtree at level {levels[level_idx]!r}; expected dict or list"
        )

    normalized: Dict[str, Any] = {}
    for top_name, subtree in tree_top_to_bottom.items():
        if not isinstance(top_name, str):
            raise TypeError("Hierarchy node names must be strings")
        normalized[top_name] = ensure(top_idx, subtree, [top_name])

    return normalized


def build_hierarchy(tree: Dict, levels: Sequence[str]) -> Hierarchy:
    """Build an N-level hierarchy from a uniform top->bottom nested ``tree``.

    Use :func:`pad_ragged_tree` before this function if your input skips levels.
    """

    levels = list(levels)
    if not levels:
        raise ValueError("levels must contain at least one level name")
    if not isinstance(tree, dict):
        raise TypeError("tree must be a dict mapping parent names to child subtrees")
    if not tree:
        raise ValueError("tree must contain at least one entry")

    names: Dict[str, List[str]] = {level: [] for level in levels}
    edges: Dict[Tuple[str, str], List[Tuple[int, int]]] = {
        (levels[idx], levels[idx + 1]): [] for idx in range(len(levels) - 1)
    }

    def add_node(level: str, name: str) -> int:
        if not isinstance(name, str):
            raise TypeError("Hierarchy node names must be strings")
        index = len(names[level])
        names[level].append(name)
        return index

    def walk(parent_level_idx: int, parent_name: str, subtree) -> int:
        level = levels[parent_level_idx]
        parent_idx = add_node(level, parent_name)

        if parent_level_idx == 0:
            if subtree not in (None, [], {}):
                raise ValueError(
                    f"Leaf level '{level}' entries should not have child nodes; received {subtree!r}"
                )
            return parent_idx

        child_level_idx = parent_level_idx - 1
        child_level = levels[child_level_idx]

        if child_level_idx == 0:
            if not isinstance(subtree, list):
                raise TypeError("Leaf level requires a list of names")
            if len(subtree) == 0:
                raise ValueError(f"Leaf list for {parent_name!r} cannot be empty")
            seen: set[str] = set()
            for leaf_name in subtree:
                if not isinstance(leaf_name, str):
                    raise TypeError("Leaf names must be strings")
                if leaf_name in seen:
                    raise ValueError(f"Duplicate {child_level} name detected: {leaf_name!r}")
                seen.add(leaf_name)
                leaf_idx = add_node(child_level, leaf_name)
                edges[(child_level, level)].append((leaf_idx, parent_idx))
            return parent_idx

        if not isinstance(subtree, dict):
            raise TypeError(f"Internal level {level!r} requires a dict of children")
        if not subtree:
            raise ValueError(f"{level!r} entry {parent_name!r} must contain at least one child")

        for child_name, child_subtree in subtree.items():
            child_idx = walk(child_level_idx, child_name, child_subtree)
            edges[(child_level, level)].append((child_idx, parent_idx))

        return parent_idx

    for top_name, subtree in tree.items():
        walk(len(levels) - 1, top_name, subtree)

    maps_adjacent: Dict[Tuple[str, str], np.ndarray] = {}
    index_adjacent: Dict[Tuple[str, str], np.ndarray] = {}
    for idx in range(len(levels) - 1):
        child_level, parent_level = levels[idx], levels[idx + 1]
        n_child = len(names[child_level])
        n_parent = len(names[parent_level])
        mapping = np.zeros((n_child, n_parent), dtype=float)
        parent_indices = np.zeros(n_child, dtype=int)

        for child_idx, parent_idx in edges[(child_level, parent_level)]:
            mapping[child_idx, parent_idx] = 1.0
            parent_indices[child_idx] = parent_idx

        if not np.all(mapping.sum(axis=1) == 1.0):
            raise ValueError(f"Invalid hierarchy: each {child_level} must map to exactly one {parent_level}")

        maps_adjacent[(child_level, parent_level)] = mapping
        index_adjacent[(child_level, parent_level)] = parent_indices

    hierarchy = Hierarchy(
        levels=levels,
        names=names,
        maps_adjacent=maps_adjacent,
        index_adjacent=index_adjacent,
    )
    hierarchy.__dict__.update(_compat_three_level_fields(hierarchy))
    return hierarchy
