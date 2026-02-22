"""Execute all notebooks in the `./examples directory."""
from pathlib import Path

import papermill as pm

for nb in Path("./examples").glob("*.ipynb"):
    pm.execute_notebook(
        input_path=nb,
        output_path=nb  # Path to save executed notebook
    )