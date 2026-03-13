"""Monte Carlo simulation for percolation threshold estimation."""

import numpy as np
from numpy.typing import NDArray

from percolation.sticks import generate_sticks, find_intersections
from percolation.clusters import find_percolating_cluster


def percolation_probability(
    eta: float,
    domain_size: float = 10.0,
    stick_length: float = 1.0,
    n_trials: int = 100,
    direction: str = "both",
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate percolation probability at dimensionless density η = N·L²/A.

    Parameters
    ----------
    eta : float
        Dimensionless stick density η = n_sticks * L² / domain_size².
    domain_size : float
        Side length of square domain (in units of stick_length).
    stick_length : float
        Length of each stick.
    n_trials : int
        Number of Monte Carlo trials.
    direction : str
        Percolation direction: "x", "y", or "both".
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
        pairs = find_intersections(sticks)
        if find_percolating_cluster(
            n_sticks, pairs, sticks, domain_size, direction
        ):
            n_perc += 1

    return n_perc / n_trials


def estimate_threshold(
    eta_range: tuple[float, float] = (4.0, 8.0),
    n_points: int = 20,
    domain_size: float = 10.0,
    stick_length: float = 1.0,
    n_trials: int = 100,
    direction: str = "both",
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Estimate percolation threshold by scanning η and finding P=0.5.

    Parameters
    ----------
    eta_range : tuple of float
        (eta_min, eta_max) range to scan.
    n_points : int
        Number of η values to sample.
    domain_size : float
        Side length of square domain.
    stick_length : float
        Stick length.
    n_trials : int
        MC trials per η value.
    direction : str
        Percolation direction.
    rng : numpy.random.Generator, optional

    Returns
    -------
    etas : ndarray
        Array of η values.
    probs : ndarray
        Percolation probabilities.
    eta_c : float
        Estimated threshold (linear interpolation at P=0.5).
    """
    if rng is None:
        rng = np.random.default_rng()

    etas = np.linspace(eta_range[0], eta_range[1], n_points)
    probs = np.zeros(n_points)

    for i, eta in enumerate(etas):
        probs[i] = percolation_probability(
            eta, domain_size, stick_length, n_trials, direction, rng
        )

    # Interpolate to find P=0.5 crossing
    eta_c = float(np.interp(0.5, probs, etas))

    return etas, probs, eta_c
