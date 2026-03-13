"""Crack network generation: parallel slits and Voronoi tessellation."""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Voronoi


def generate_parallel_cracks(
    n_cracks: int,
    domain_size: float = 1.0,
    crack_width: float = 0.01,
    orientation: str = "vertical",
    rng: np.random.Generator | None = None,
) -> list[NDArray[np.float64]]:
    """Generate parallel straight cracks (as in Baret et al. 2024).

    Each crack is represented as a polygon (rectangle).

    Parameters
    ----------
    n_cracks : int
        Number of parallel cracks.
    domain_size : float
        Side length of domain.
    crack_width : float
        Width of each crack.
    orientation : str
        "vertical" or "horizontal".
    rng : numpy.random.Generator, optional

    Returns
    -------
    cracks : list of ndarray, shape (4, 2)
        Each crack as a rectangle polygon.
    crack_segments : not returned here — use crack_segments_from_parallel().
    """
    if rng is None:
        rng = np.random.default_rng()

    # Uniformly space crack centers
    positions = np.linspace(
        domain_size / (n_cracks + 1),
        domain_size * n_cracks / (n_cracks + 1),
        n_cracks,
    )

    cracks = []
    hw = crack_width / 2

    for pos in positions:
        if orientation == "vertical":
            rect = np.array([
                [pos - hw, 0.0],
                [pos + hw, 0.0],
                [pos + hw, domain_size],
                [pos - hw, domain_size],
            ])
        else:
            rect = np.array([
                [0.0, pos - hw],
                [domain_size, pos - hw],
                [domain_size, pos + hw],
                [0.0, pos + hw],
            ])
        cracks.append(rect)

    return cracks


def crack_segments_from_parallel(
    n_cracks: int,
    domain_size: float = 1.0,
    orientation: str = "vertical",
) -> list[tuple[NDArray, NDArray]]:
    """Return crack center lines as segments for intersection testing.

    Parameters
    ----------
    n_cracks : int
    domain_size : float
    orientation : str

    Returns
    -------
    segments : list of (p1, p2) endpoint pairs
    """
    positions = np.linspace(
        domain_size / (n_cracks + 1),
        domain_size * n_cracks / (n_cracks + 1),
        n_cracks,
    )
    segments = []
    for pos in positions:
        if orientation == "vertical":
            segments.append((
                np.array([pos, 0.0]),
                np.array([pos, domain_size]),
            ))
        else:
            segments.append((
                np.array([0.0, pos]),
                np.array([domain_size, pos]),
            ))
    return segments


def generate_voronoi_cracks(
    n_seeds: int,
    domain_size: float = 1.0,
    rng: np.random.Generator | None = None,
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Generate a Voronoi crack network.

    Seeds are placed randomly in the domain. The Voronoi edges
    (ridges) represent cracks. Only edges fully within the domain
    are kept.

    Parameters
    ----------
    n_seeds : int
        Number of Voronoi seeds.
    domain_size : float
        Side length of domain.
    rng : numpy.random.Generator, optional

    Returns
    -------
    crack_segments : list of (p1, p2)
        Each crack as a line segment (two endpoints).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate seed points
    points = rng.uniform(0, domain_size, (n_seeds, 2))

    # Mirror points for better boundary behavior
    mirrored = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = points.copy()
        shifted[:, 0] += dx * domain_size
        shifted[:, 1] += dy * domain_size
        mirrored.append(shifted)
    all_points = np.vstack([points] + mirrored)

    vor = Voronoi(all_points)

    margin = -0.01 * domain_size  # small negative margin to include boundary edges
    segments = []
    for ridge_verts in vor.ridge_vertices:
        if -1 in ridge_verts:
            continue  # skip infinite ridges
        v0 = vor.vertices[ridge_verts[0]]
        v1 = vor.vertices[ridge_verts[1]]

        # Keep only edges with both vertices inside domain (with margin)
        if (margin <= v0[0] <= domain_size - margin
                and margin <= v0[1] <= domain_size - margin
                and margin <= v1[0] <= domain_size - margin
                and margin <= v1[1] <= domain_size - margin):
            segments.append((v0.copy(), v1.copy()))

    return segments


def sticks_bridging_crack(
    sticks: NDArray[np.float64],
    crack_segment: tuple[NDArray, NDArray],
) -> NDArray[np.bool_]:
    """Determine which sticks bridge (cross) a given crack segment.

    A stick bridges the crack if the stick segment intersects
    the crack segment.

    Parameters
    ----------
    sticks : ndarray, shape (n, 2, 2)
    crack_segment : (p1, p2)

    Returns
    -------
    bridging : bool array, shape (n,)
    """
    from percolation.sticks import segments_intersect

    p3, p4 = crack_segment
    n = len(sticks)

    # Broadcast crack segment
    p3_arr = np.broadcast_to(p3, (n, 2))
    p4_arr = np.broadcast_to(p4, (n, 2))

    return segments_intersect(sticks[:, 0], sticks[:, 1], p3_arr, p4_arr)


def count_bridging_sticks(
    sticks: NDArray[np.float64],
    crack_segments: list[tuple[NDArray, NDArray]],
) -> int:
    """Count total number of sticks that bridge at least one crack."""
    bridging = np.zeros(len(sticks), dtype=bool)
    for seg in crack_segments:
        bridging |= sticks_bridging_crack(sticks, seg)
    return int(np.sum(bridging))
