"""Tests for stick generation and intersection detection."""

import numpy as np
import pytest
from percolation.sticks import generate_sticks, find_intersections, segments_intersect


def test_generate_sticks_shape():
    sticks = generate_sticks(100, length=1.0, domain_size=10.0)
    assert sticks.shape == (100, 2, 2)


def test_generate_sticks_length():
    rng = np.random.default_rng(42)
    sticks = generate_sticks(1000, length=2.0, domain_size=10.0, rng=rng)
    lengths = np.sqrt(np.sum((sticks[:, 1] - sticks[:, 0]) ** 2, axis=1))
    np.testing.assert_allclose(lengths, 2.0, atol=1e-12)


def test_segments_intersect_cross():
    # Two crossing segments
    p1, p2 = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    p3, p4 = np.array([0.0, 1.0]), np.array([1.0, 0.0])
    assert segments_intersect(p1, p2, p3, p4)


def test_segments_intersect_parallel():
    # Parallel segments
    p1, p2 = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    p3, p4 = np.array([0.0, 1.0]), np.array([1.0, 1.0])
    assert not segments_intersect(p1, p2, p3, p4)


def test_segments_intersect_no_overlap():
    # Non-parallel but segments don't reach each other
    p1, p2 = np.array([0.0, 0.0]), np.array([0.3, 0.3])
    p3, p4 = np.array([0.7, 0.0]), np.array([1.0, 1.0])
    assert not segments_intersect(p1, p2, p3, p4)


def test_find_intersections_simple():
    # Two crossing sticks
    sticks = np.array([
        [[0.0, 0.5], [1.0, 0.5]],  # horizontal
        [[0.5, 0.0], [0.5, 1.0]],  # vertical
    ])
    pairs = find_intersections(sticks)
    assert len(pairs) == 1
    assert pairs[0] == (0, 1)


def test_find_intersections_none():
    # Two parallel sticks
    sticks = np.array([
        [[0.0, 0.0], [1.0, 0.0]],
        [[0.0, 1.0], [1.0, 1.0]],
    ])
    pairs = find_intersections(sticks)
    assert len(pairs) == 0


def test_find_intersections_many():
    rng = np.random.default_rng(123)
    sticks = generate_sticks(50, length=1.0, domain_size=5.0, rng=rng)
    pairs = find_intersections(sticks)
    # At η = 50*1/25 = 2.0, should have some intersections but not too many
    assert len(pairs) > 0
