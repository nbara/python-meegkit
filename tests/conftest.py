import pytest
import numpy as np
import random as rand


@pytest.fixture
def random():
    rand.seed(9)
    np.random.seed(9)
