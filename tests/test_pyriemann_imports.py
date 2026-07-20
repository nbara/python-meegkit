"""pyriemann API compatibility tests."""
from pathlib import Path

import numpy as np

from meegkit.utils.covariances import block_covariance

ROOT = Path(__file__).resolve().parents[1]


def _version_tuple(version):
    return tuple(int(part) for part in version.split("."))


def test_pyriemann_lower_bound_uses_geometry_api():
    """Ensure the supported pyriemann range has the geometry namespace."""
    requirement = next(
        line
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.startswith("pyriemann")
    )

    _, lower_bound = requirement.split(">=")

    assert _version_tuple(lower_bound) >= (0, 12)


def test_pyriemann_imports_use_modern_public_api():
    """Keep deprecated pyriemann utils imports out of meegkit."""
    source_files = [
        ROOT / "meegkit" / "asr.py",
        ROOT / "meegkit" / "trca.py",
        ROOT / "meegkit" / "utils" / "covariances.py",
    ]
    source = "\n".join(path.read_text() for path in source_files)

    assert "pyriemann.utils.mean" not in source
    assert "pyriemann.utils.covariance" not in source
    assert "check_function" not in source
    assert "cov_est_functions" not in source


def test_block_covariance_uses_pyriemann_covariances():
    """Compare block covariance with pyriemann's vectorized covariance helper."""
    from pyriemann.geometry.covariance import covariances

    data = 0.1 + 0.01 * (1 + np.arange(24, dtype=float).reshape(3, 8))
    window = 4
    overlap = 0.5
    jump = max(int(round(window * (1 - overlap))), 1)
    n_samples = data.shape[1]
    expected_blocks = np.array(
        [data[:, start:start + window] for start in range(0, n_samples - window + 1, jump)]
    )

    actual = block_covariance(data, window=window, overlap=overlap, padding=False)

    np.testing.assert_allclose(actual, covariances(expected_blocks, estimator="cov"))
