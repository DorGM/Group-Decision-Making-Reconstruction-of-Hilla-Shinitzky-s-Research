"""Seeding utility. Plan §8 OQ15: "Seeds fixed and recorded (0)"."""
import random

import numpy as np

DEFAULT_SEED = 0


def set_seed(seed=DEFAULT_SEED):
    """Seed Python's and numpy's global generators; return a numpy Generator.

    Use the returned Generator for new code so the seed is explicit.
    """
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
