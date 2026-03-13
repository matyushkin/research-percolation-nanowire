"""Tests for crack generation and bridge percolation."""

import numpy as np
from percolation.cracks import (
    crack_segments_from_parallel,
    generate_voronoi_cracks,
    sticks_bridging_crack,
    count_bridging_sticks,
)
from percolation.bridge import bridge_percolation


def test_parallel_cracks():
    segs = crack_segments_from_parallel(3, domain_size=10.0, orientation="vertical")
    assert len(segs) == 3
    # All segments should be vertical (same x for both endpoints)
    for p1, p2 in segs:
        assert abs(p1[0] - p2[0]) < 1e-10


def test_voronoi_cracks():
    rng = np.random.default_rng(42)
    segs = generate_voronoi_cracks(20, domain_size=10.0, rng=rng)
    assert len(segs) > 0
    # All endpoints should be within domain
    for p1, p2 in segs:
        assert -0.2 <= p1[0] <= 10.2
        assert -0.2 <= p1[1] <= 10.2


def test_bridging_detection():
    # Horizontal stick crossing a vertical crack at x=5
    sticks = np.array([
        [[4.0, 5.0], [6.0, 5.0]],  # crosses x=5
        [[1.0, 5.0], [3.0, 5.0]],  # doesn't cross x=5
    ])
    crack = (np.array([5.0, 0.0]), np.array([5.0, 10.0]))
    bridging = sticks_bridging_crack(sticks, crack)
    assert bridging[0] == True
    assert bridging[1] == False


def test_bridge_percolation_no_cracks():
    # Without cracks, should behave like standard percolation
    rng = np.random.default_rng(42)
    # High density → should percolate
    from percolation.sticks import generate_sticks
    sticks = generate_sticks(800, length=1.0, domain_size=10.0, rng=rng)
    result = bridge_percolation(sticks, [], domain_size=10.0, direction="x")
    assert result is True


def test_bridge_percolation_with_crack():
    # One vertical crack at x=5, high density of sticks → should bridge
    rng = np.random.default_rng(42)
    from percolation.sticks import generate_sticks
    cracks = crack_segments_from_parallel(1, domain_size=10.0, orientation="vertical")
    sticks = generate_sticks(800, length=1.0, domain_size=10.0, rng=rng)
    result = bridge_percolation(sticks, cracks, domain_size=10.0, direction="x")
    assert result is True


def test_bridge_percolation_low_density():
    # Very low density with crack → should not percolate
    rng = np.random.default_rng(42)
    from percolation.sticks import generate_sticks
    cracks = crack_segments_from_parallel(3, domain_size=10.0, orientation="vertical")
    sticks = generate_sticks(5, length=1.0, domain_size=10.0, rng=rng)
    result = bridge_percolation(sticks, cracks, domain_size=10.0, direction="x")
    assert result is False
