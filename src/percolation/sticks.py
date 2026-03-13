"""Random stick generation and intersection detection."""

import numpy as np
from numpy.typing import NDArray


def generate_sticks(
    n_sticks: int,
    length: float = 1.0,
    domain_size: float = 1.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Generate random sticks (line segments) in a 2D square domain.

    Sticks are placed with centers uniformly distributed in the domain
    and orientations uniformly distributed in [0, π).

    Parameters
    ----------
    n_sticks : int
        Number of sticks to generate.
    length : float
        Length of each stick.
    domain_size : float
        Side length of the square domain.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    sticks : ndarray, shape (n_sticks, 2, 2)
        Array of endpoints: sticks[i] = [[x0, y0], [x1, y1]].
    """
    if rng is None:
        rng = np.random.default_rng()

    # Center positions (uniform in domain)
    cx = rng.uniform(0, domain_size, n_sticks)
    cy = rng.uniform(0, domain_size, n_sticks)

    # Orientations (uniform in [0, π))
    theta = rng.uniform(0, np.pi, n_sticks)

    half_l = length / 2.0
    dx = half_l * np.cos(theta)
    dy = half_l * np.sin(theta)

    sticks = np.empty((n_sticks, 2, 2))
    sticks[:, 0, 0] = cx - dx
    sticks[:, 0, 1] = cy - dy
    sticks[:, 1, 0] = cx + dx
    sticks[:, 1, 1] = cy + dy

    return sticks


def _cross2d(a: NDArray, b: NDArray) -> NDArray:
    """2D cross product of vectors a and b."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def segments_intersect(
    p1: NDArray, p2: NDArray, p3: NDArray, p4: NDArray
) -> NDArray[np.bool_]:
    """Test intersection of segments (p1,p2) with (p3,p4).

    Uses the cross-product orientation method.
    Works element-wise on arrays of segments.
    """
    d1 = p2 - p1  # direction of segment 1
    d2 = p4 - p3  # direction of segment 2

    cross_d = _cross2d(d1, d2)

    # Parallel segments (cross ~ 0) are treated as non-intersecting
    parallel = np.abs(cross_d) < 1e-12

    # Parameters t, u for intersection point
    d3 = p3 - p1
    t = _cross2d(d3, d2)
    u = _cross2d(d3, d1)

    # Avoid division by zero for parallel segments
    safe_cross = np.where(parallel, 1.0, cross_d)
    t = t / safe_cross
    u = u / safe_cross

    return (~parallel) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)


def find_intersections(
    sticks: NDArray[np.float64],
    cell_size: float | None = None,
) -> list[tuple[int, int]]:
    """Find all pairs of intersecting sticks using spatial hashing.

    Parameters
    ----------
    sticks : ndarray, shape (n, 2, 2)
        Array of stick endpoints.
    cell_size : float, optional
        Grid cell size for spatial hashing. Defaults to stick length.

    Returns
    -------
    pairs : list of (int, int)
        Pairs of indices of intersecting sticks.
    """
    n = len(sticks)
    if n < 2:
        return []

    # Estimate stick length from first stick
    if cell_size is None:
        d = sticks[0, 1] - sticks[0, 0]
        cell_size = float(np.sqrt(d[0] ** 2 + d[1] ** 2))
        if cell_size < 1e-12:
            cell_size = 1.0

    # Build spatial hash grid
    grid: dict[tuple[int, int], list[int]] = {}

    for i in range(n):
        x0, y0 = sticks[i, 0]
        x1, y1 = sticks[i, 1]

        # Cells covered by this stick's bounding box
        c_x0 = int(np.floor(min(x0, x1) / cell_size))
        c_x1 = int(np.floor(max(x0, x1) / cell_size))
        c_y0 = int(np.floor(min(y0, y1) / cell_size))
        c_y1 = int(np.floor(max(y0, y1) / cell_size))

        for cx in range(c_x0, c_x1 + 1):
            for cy in range(c_y0, c_y1 + 1):
                cell = (cx, cy)
                if cell not in grid:
                    grid[cell] = []
                grid[cell].append(i)

    # Check intersection for candidate pairs
    pairs = []
    checked: set[tuple[int, int]] = set()

    for cell_sticks in grid.values():
        for ii in range(len(cell_sticks)):
            for jj in range(ii + 1, len(cell_sticks)):
                i, j = cell_sticks[ii], cell_sticks[jj]
                pair = (min(i, j), max(i, j))
                if pair in checked:
                    continue
                checked.add(pair)

                if segments_intersect(
                    sticks[i, 0], sticks[i, 1],
                    sticks[j, 0], sticks[j, 1],
                ):
                    pairs.append(pair)

    return pairs
