"""Solve microscope design curves and validate exported NPZ.

Usage:
    python examples/microscope_models/solve_design_curves.py
"""

from pathlib import Path
import runpy
import numpy as np


def main() -> None:
    script_path = Path("examples/microscope_models/microscope_design_curves.py")
    runpy.run_path(str(script_path), run_name="__main__")

    export_path = Path("examples/microscope_models/data/microscope_design_curves.npz")
    if not export_path.exists():
        raise FileNotFoundError(f"Design curves export not found: {export_path}")

    required_keys = {
        "spot_control_values",
        "spot_lens_focals_m",
        "spot_currents_a",
        "mag_control_values",
        "mag_lens_focals_m",
        "mag_currents_a_il",
        "mag_currents_a_comp",
    }
    with np.load(export_path) as data:
        missing = sorted(required_keys.difference(set(data.files)))
        if missing:
            raise KeyError(f"Missing NPZ keys: {missing}")
        print(f"Design curves ready: {export_path}")
        print(f"Keys: {sorted(data.files)}")


if __name__ == "__main__":
    main()
