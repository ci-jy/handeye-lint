import numpy as np
import pytest

from handeye import sim


@pytest.fixture
def clean_eih():
    return sim.generate(sim.ScenarioConfig(rot_noise_deg=0, trans_noise_mm=0),
                        np.random.default_rng(7))
