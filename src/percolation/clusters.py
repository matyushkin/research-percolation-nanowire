"""Union-Find data structure and percolation cluster analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class UnionFind:
    """Weighted Union-Find with path compression."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def cluster_size(self, x: int) -> int:
        return self.size[self.find(x)]


def find_percolating_cluster(
    n_sticks: int,
    pairs: list[tuple[int, int]],
    sticks: NDArray[np.float64],
    domain_size: float,
    direction: str = "x",
) -> bool:
    """Check if a percolating cluster spans the domain.

    A cluster percolates if it contains sticks touching both
    opposite boundaries of the domain.

    Parameters
    ----------
    n_sticks : int
        Total number of sticks.
    pairs : list of (int, int)
        Pairs of intersecting stick indices.
    sticks : ndarray, shape (n, 2, 2)
        Stick endpoints.
    domain_size : float
        Side length of the square domain.
    direction : str
        "x" to check left-right spanning, "y" for top-bottom,
        "both" to require spanning in both directions.

    Returns
    -------
    percolates : bool
    """
    uf = UnionFind(n_sticks)
    for i, j in pairs:
        uf.union(i, j)

    axis_map = {"x": 0, "y": 1}

    def _check_direction(axis: int) -> bool:
        # Find sticks touching low boundary (coord < 0)
        low_mask = np.minimum(sticks[:, 0, axis], sticks[:, 1, axis]) <= 0
        # Find sticks touching high boundary (coord > domain_size)
        high_mask = (
            np.maximum(sticks[:, 0, axis], sticks[:, 1, axis]) >= domain_size
        )

        low_indices = np.where(low_mask)[0]
        high_indices = np.where(high_mask)[0]

        if len(low_indices) == 0 or len(high_indices) == 0:
            return False

        # Get cluster roots for boundary sticks
        low_roots = {uf.find(int(i)) for i in low_indices}
        for j in high_indices:
            if uf.find(int(j)) in low_roots:
                return True
        return False

    if direction == "both":
        return _check_direction(0) and _check_direction(1)
    else:
        return _check_direction(axis_map[direction])
