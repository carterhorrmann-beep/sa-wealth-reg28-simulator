"""Shared fixtures. Everything runs offline and in a few seconds."""

import numpy as np
import pytest

from sa_wealth.assets import default_assumptions
from sa_wealth.montecarlo import SimulationConfig, generate_paths
from sa_wealth.reg28_tax import STATIC_PRESETS


@pytest.fixture(scope="session")
def assumptions():
    return default_assumptions()


@pytest.fixture(scope="session")
def moderate():
    return np.array([STATIC_PRESETS["Moderate (Reg 28)"][a] for a in default_assumptions().assets])


@pytest.fixture(scope="session")
def config():
    return SimulationConfig(initial_lump_sum=250_000, monthly_contribution=5_000, horizon_years=10, target_real=1_500_000, n_paths=1_500, seed=7)


@pytest.fixture(scope="session")
def paths(assumptions, config):
    return generate_paths(assumptions, config.n_paths, config.horizon_years, config.seed)
