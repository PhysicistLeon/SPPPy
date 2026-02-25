"""Tests for global M_matrix cache controls."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SPPPy import clear_m_cache, get_m_cache_limit, get_m_cache_size, set_m_cache_limit
from SPPPy.materials import M_matrix


def test_m_cache_limit_validation():
    """set_m_cache_limit must reject non-positive values."""
    try:
        set_m_cache_limit(0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-positive cache limit")


def test_m_cache_clear_on_limit_policy():
    """Cache should be cleared when insertion reaches configured limit."""
    old_limit = get_m_cache_limit()
    try:
        clear_m_cache()
        set_m_cache_limit(2)

        M_matrix(1.0, 0.1, 1.5, 1.0, 0.0, "p")
        assert get_m_cache_size() == 1

        M_matrix(1.1, 0.1, 1.5, 1.0, 0.0, "p")
        assert get_m_cache_size() == 2

        M_matrix(1.2, 0.1, 1.5, 1.0, 0.0, "p")
        assert get_m_cache_size() == 1
    finally:
        set_m_cache_limit(old_limit)
        clear_m_cache()
