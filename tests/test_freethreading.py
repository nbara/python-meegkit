import subprocess
import sys
import sysconfig

import pytest

pytestmark = pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="requires a free-threaded CPython build",
)


def test_import_does_not_reenable_gil():
    # test dependencies (such as pytest extensions) could re-enable the GIL
    # without making the library non-freethreaded so we need to run this in
    # its own subprocess.
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, meegkit; raise SystemExit(int(sys._is_gil_enabled()))",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
