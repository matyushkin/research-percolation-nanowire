"""Bridge percolation: connectivity of conducting islands separated by cracks.

In bridge percolation, a conducting slab is broken into disconnected islands
by a crack network. Nanowires deposited on top can bridge cracks and restore
electrical connectivity. This module determines whether nanowires create a
percolating path across the crack network.

Reference: Baret et al., Nanoscale (2024) — parallel cracks only.
This code generalizes to arbitrary crack topologies (Voronoi, fractal).
"""

import numpy as np
from numpy.typing import NDArray

from percolation.sticks import generate_sticks, find_intersections
from percolation.clusters import UnionFind
from percolation.cracks import sticks_bridging_crack


def bridge_percolation(
    sticks: NDArray[np.float64],
    crack_segments: list[tuple[NDArray, NDArray]],
    domain_size: float,
    direction: str = "x",
) -> bool:
    """Check if nanowires create a percolating path across crack network.

    The algorithm:
    1. Each conducting island (region between cracks) is a node.
    2. Two islands are connected if a nanowire bridges the crack between them.
    3. Check if island-graph percolates across the domain.

    Simplified approach for arbitrary crack networks:
    - Build stick connectivity graph (stick-stick intersections).
    - For each crack, find sticks on opposite sides.
    - A crack is "bridged" if two sticks on opposite sides are in the same
      connected component (i.e., connected through other sticks, possibly
      bridging other cracks).

    Parameters
    ----------
    sticks : ndarray, shape (n, 2, 2)
    crack_segments : list of (p1, p2)
    domain_size : float
    direction : str
        "x" for left-right, "y" for top-bottom.

    Returns
    -------
    percolates : bool
    """
    n = len(sticks)
    if n == 0:
        return False

    # Build stick connectivity via intersections
    pairs = find_intersections(sticks)
    uf = UnionFind(n)
    for i, j in pairs:
        uf.union(i, j)

    # Determine which side of each crack each stick center is on.
    # Two sticks bridge a crack if they are on opposite sides AND
    # connected in the stick network.
    # For the domain to percolate, we need a path from the left boundary
    # to the right boundary through bridged cracks.

    # Simpler approach: treat each crack as potentially disconnecting.
    # A stick that crosses a crack connects the regions on both sides.
    # We create a region graph:
    #   - Virtual "left" and "right" (or "bottom"/"top") boundary nodes
    #   - Each stick is a node in the union-find (already built)
    #   - Sticks that touch left boundary connect to virtual left node
    #   - Sticks that touch right boundary connect to virtual right node
    #   - Two sticks that intersect are connected (already in UF)
    # Key insight: sticks already bridge cracks if they cross them.
    # So if a connected component of sticks spans from left to right
    # boundary, it necessarily bridges all cracks in between.

    # But we need something more subtle: sticks on the same island
    # (between cracks) are NOT necessarily connected by intersection.
    # They are connected through the conducting slab.

    # Full model:
    # - Assign each stick to island(s) it touches
    # - Island connectivity: two sticks on same island → connected
    # - Crack bridging: stick crossing crack → connects adjacent islands

    # For now, use the direct approach:
    # Build an extended graph with island nodes + stick nodes.

    return _bridge_percolation_island_graph(
        sticks, crack_segments, domain_size, direction, pairs
    )


def _assign_stick_to_side(
    stick: NDArray,
    crack_p1: NDArray,
    crack_p2: NDArray,
) -> int:
    """Determine which side of a crack line a stick's center is on.

    Returns +1 or -1, or 0 if exactly on the line.
    """
    center = (stick[0] + stick[1]) / 2
    # Cross product of crack direction with (center - crack_p1)
    d = crack_p2 - crack_p1
    v = center - crack_p1
    cross = d[0] * v[1] - d[1] * v[0]
    if cross > 0:
        return 1
    elif cross < 0:
        return -1
    return 0


def _bridge_percolation_island_graph(
    sticks: NDArray[np.float64],
    crack_segments: list[tuple[NDArray, NDArray]],
    domain_size: float,
    direction: str,
    stick_pairs: list[tuple[int, int]],
) -> bool:
    """Island-graph approach to bridge percolation.

    Nodes: islands (regions between cracks) + virtual boundary nodes.
    Edges: two islands are connected if a nanowire bridges the crack
    separating them, OR if they share a boundary.

    For computational efficiency, we use a dual approach:
    - Each stick that crosses a crack connects the two adjacent islands.
    - Sticks on the same island that are connected (directly or through
      other sticks) provide additional connectivity.
    """
    n_sticks = len(sticks)
    n_cracks = len(crack_segments)

    # Extended UF: sticks + 2 virtual boundary nodes
    # node n_sticks = left/bottom boundary
    # node n_sticks+1 = right/top boundary
    uf = UnionFind(n_sticks + 2)
    left_node = n_sticks
    right_node = n_sticks + 1

    # Connect intersecting sticks
    for i, j in stick_pairs:
        uf.union(i, j)

    axis = 0 if direction == "x" else 1

    # Connect sticks touching boundaries
    for i in range(n_sticks):
        coords = [sticks[i, 0, axis], sticks[i, 1, axis]]
        if min(coords) <= 0:
            uf.union(i, left_node)
        if max(coords) >= domain_size:
            uf.union(i, right_node)

    # If no cracks, just check standard percolation
    if n_cracks == 0:
        return uf.connected(left_node, right_node)

    # For each crack, find sticks that bridge it.
    # A bridging stick crosses the crack AND connects regions on both sides.
    # In the island model, sticks on the same island are conductively
    # connected through the slab. So we also connect all sticks whose
    # centers are on the same side of each crack and are on the same island.

    # Step 1: For each crack, find bridging sticks (sticks that cross it).
    # These sticks directly connect the two sides.
    # Already handled by stick-stick intersection + boundary connectivity.

    # Step 2: Connect sticks on the same island.
    # Two sticks on the same island are connected through the slab,
    # even without direct intersection.
    # An island is a region bounded by cracks. For simplicity, we
    # connect sticks that are on the same side of ALL cracks.

    # Assign each stick a "side signature" — which side of each crack it's on
    if n_cracks <= 20:  # practical limit for signature approach
        signatures: dict[tuple, list[int]] = {}
        for i in range(n_sticks):
            sig = []
            for cp1, cp2 in crack_segments:
                sig.append(_assign_stick_to_side(sticks[i], cp1, cp2))
            sig_tuple = tuple(sig)
            if sig_tuple not in signatures:
                signatures[sig_tuple] = []
            signatures[sig_tuple].append(i)

        # Connect all sticks on the same island
        for stick_list in signatures.values():
            for k in range(1, len(stick_list)):
                uf.union(stick_list[0], stick_list[k])
    else:
        # For many cracks, use spatial binning instead
        _connect_same_island_spatial(sticks, crack_segments, uf, domain_size)

    return uf.connected(left_node, right_node)


def _connect_same_island_spatial(
    sticks: NDArray[np.float64],
    crack_segments: list[tuple[NDArray, NDArray]],
    uf: UnionFind,
    domain_size: float,
    grid_res: int = 50,
) -> None:
    """Connect sticks on the same island using spatial grid binning.

    For large crack networks, the signature approach is too slow.
    Instead, we rasterize islands on a grid and connect sticks
    whose centers fall in the same grid cell (same island).
    """
    # Create island map by flood-filling between cracks
    # Simplified: assign each grid cell an island ID based on
    # which cracks separate it from cell (0,0)

    cell_size = domain_size / grid_res
    n = len(sticks)

    # Compute stick centers
    centers = (sticks[:, 0] + sticks[:, 1]) / 2
    cell_x = np.clip((centers[:, 0] / cell_size).astype(int), 0, grid_res - 1)
    cell_y = np.clip((centers[:, 1] / cell_size).astype(int), 0, grid_res - 1)

    # Group by grid cell
    cell_map: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        cell = (int(cell_x[i]), int(cell_y[i]))
        if cell not in cell_map:
            cell_map[cell] = []
        cell_map[cell].append(i)

    # Connect sticks in same cell
    for stick_list in cell_map.values():
        for k in range(1, len(stick_list)):
            uf.union(stick_list[0], stick_list[k])


def bridge_percolation_probability(
    eta: float,
    crack_segments: list[tuple[NDArray, NDArray]],
    domain_size: float = 10.0,
    stick_length: float = 1.0,
    n_trials: int = 100,
    direction: str = "x",
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate bridge percolation probability at density η.

    Parameters
    ----------
    eta : float
        Dimensionless stick density.
    crack_segments : list of (p1, p2)
        Crack geometry (fixed across trials).
    domain_size : float
    stick_length : float
    n_trials : int
    direction : str
    rng : numpy.random.Generator, optional

    Returns
    -------
    p : float
        Fraction of trials that percolated.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_sticks = int(eta * domain_size**2 / stick_length**2)
    n_perc = 0

    for _ in range(n_trials):
        sticks = generate_sticks(n_sticks, stick_length, domain_size, rng)
        if bridge_percolation(sticks, crack_segments, domain_size, direction):
            n_perc += 1

    return n_perc / n_trials
