"""Hierarchy construction utilities for the Precision MMM package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


@dataclass(init=False)
class Hierarchy:
    """Representation of an arbitrary N-level hierarchy."""

    levels: List[str]
    names: Dict[str, List[str]]
    maps_adjacent: Dict[Tuple[str, str], np.ndarray]
    index_adjacent: Dict[Tuple[str, str], np.ndarray]

    def __init__(
        self,
        *,
        levels: Sequence[str] | None = None,
        names: Dict[str, Sequence[str]] | None = None,
        maps_adjacent: Dict[Tuple[str, str], np.ndarray] | None = None,
        index_adjacent: Dict[Tuple[str, str], np.ndarray] | None = None,
        **legacy_kwargs: Any,
    ) -> None:
        """Support both modern and legacy construction signatures.

        The modern API constructs a :class:`Hierarchy` from explicit ``levels``,
        ``names``, ``maps_adjacent``, and ``index_adjacent`` arguments (used by
        :func:`build_hierarchy`).  Older code paths instantiated ``Hierarchy``
        directly by passing channel / platform / tactical metadata and mapping
        matrices.  To keep those workflows working we translate the legacy
        arguments into the new representation here.
        """

        if (
            levels is not None
            or names is not None
            or maps_adjacent is not None
            or index_adjacent is not None
        ):
            if None in (levels, names, maps_adjacent, index_adjacent):
                raise TypeError(
                    "Hierarchy() requires levels, names, maps_adjacent, and index_adjacent when using the structured API"
                )
            if legacy_kwargs:
                extra = ", ".join(sorted(legacy_kwargs))
                raise TypeError(
                    "Hierarchy() structured API does not accept legacy keyword(s): " + extra
                )

            self.levels = list(levels or [])
            self.names = {
                level: [str(name) for name in seq]
                for level, seq in (names or {}).items()
            }
            self.maps_adjacent = {
                key: np.asarray(matrix, dtype=float)
                for key, matrix in (maps_adjacent or {}).items()
            }
            self.index_adjacent = {
                key: np.asarray(indices, dtype=int)
                for key, indices in (index_adjacent or {}).items()
            }
        else:
            self._init_from_legacy(**legacy_kwargs)

        # Simple caches for composed mapping and index lookups.
        self._map_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._index_cache: Dict[Tuple[str, str], np.ndarray] = {}
        # Populate legacy convenience attributes (M_tp, etc.) when applicable.
        self.__dict__.update(_compat_three_level_fields(self))

    def _init_from_legacy(self, **legacy_kwargs: Any) -> None:
        required = {
            "channel_names",
            "platform_names",
            "tactical_names",
            "M_tp",
            "M_tc",
            "t_to_p",
            "p_to_c",
        }
        missing = sorted(required.difference(legacy_kwargs))
        if missing:
            raise TypeError(
                "Hierarchy() legacy signature missing required arguments: "
                + ", ".join(missing)
            )

        unexpected = sorted(set(legacy_kwargs) - required)
        if unexpected:
            raise TypeError(
                "Hierarchy() legacy signature received unexpected keyword(s): "
                + ", ".join(unexpected)
            )

        tactical_names = [str(name) for name in legacy_kwargs["tactical_names"]]
        platform_names = [str(name) for name in legacy_kwargs["platform_names"]]
        channel_names = [str(name) for name in legacy_kwargs["channel_names"]]

        levels = ["tactical", "platform", "channel"]
        self.levels = levels
        self.names = {
            "tactical": tactical_names,
            "platform": platform_names,
            "channel": channel_names,
        }

        M_tp = np.asarray(legacy_kwargs["M_tp"], dtype=float)
        M_tc = np.asarray(legacy_kwargs["M_tc"], dtype=float)
        t_to_p = np.asarray(legacy_kwargs["t_to_p"], dtype=int)
        p_to_c = np.asarray(legacy_kwargs["p_to_c"], dtype=int)

        num_tactical = len(tactical_names)
        num_platform = len(platform_names)
        num_channel = len(channel_names)

        if M_tp.shape != (num_tactical, num_platform):
            raise ValueError("M_tp shape does not match tactical/platform dimensions")
        if M_tc.shape != (num_tactical, num_channel):
            raise ValueError("M_tc shape does not match tactical/channel dimensions")
        if t_to_p.shape != (num_tactical,):
            raise ValueError("t_to_p must provide a parent index for each tactical")
        if p_to_c.shape != (num_platform,):
            raise ValueError("p_to_c must provide a parent index for each platform")
        if not np.all((t_to_p >= 0) & (t_to_p < num_platform)):
            raise ValueError("t_to_p indices must be within platform range")
        if not np.all((p_to_c >= 0) & (p_to_c < num_channel)):
            raise ValueError("p_to_c indices must be within channel range")

        map_tp = M_tp.astype(float, copy=True)
        map_pc = np.zeros((num_platform, num_channel), dtype=float)
        map_pc[np.arange(num_platform), p_to_c] = 1.0

        # Validate that the implied tactical->channel mapping matches the provided M_tc.
        implied_tc = map_tp @ map_pc
        if not np.allclose(implied_tc, M_tc):
            raise ValueError("Provided M_tc is inconsistent with M_tp and p_to_c mappings")

        self.maps_adjacent = {
            ("tactical", "platform"): map_tp,
            ("platform", "channel"): map_pc,
        }
        self.index_adjacent = {
            ("tactical", "platform"): t_to_p.astype(int, copy=True),
            ("platform", "channel"): p_to_c.astype(int, copy=True),
        }

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

        key = (child, parent)
        cached = self._map_cache.get(key)
        if cached is not None:
            return cached

        result = None
        for level_idx in range(child_idx, parent_idx):
            c = self.levels[level_idx]
            p = self.levels[level_idx + 1]
            M = self.maps_adjacent[(c, p)]
            result = M if result is None else result @ M
        assert result is not None  # for mypy; parent_idx > child_idx ensured a matrix was set
        self._map_cache[key] = result
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

        key = (self.levels[child_idx], self.levels[parent_idx])
        cached = self._index_cache.get(key)
        if cached is not None:
            return cached.copy()

        indices = self.index_adjacent[(self.levels[child_idx], self.levels[child_idx + 1])].copy()
        for level_idx in range(child_idx + 1, parent_idx):
            lvl = self.levels[level_idx]
            nxt = self.levels[level_idx + 1]
            indices = self.index_adjacent[(lvl, nxt)][indices]
        self._index_cache[key] = indices.copy()
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
