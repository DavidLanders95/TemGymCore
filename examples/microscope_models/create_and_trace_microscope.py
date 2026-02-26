"""Build the microscope model from design curves and run propagation.

This delegates to `microscope_model.py`, which now writes images to:
- examples/microscope_models/output/sample_intensity.png
- examples/microscope_models/output/ray_diagram.png

Usage:
    python examples/microscope_models/create_and_trace_microscope.py
"""

from pathlib import Path
import runpy


def main() -> None:
    npz_path = Path("examples/microscope_models/data/microscope_design_curves.npz")
    if not npz_path.exists():
        raise FileNotFoundError(
            "Design-curve NPZ missing. Run solve_design_curves.py first."
        )

    runpy.run_path("examples/microscope_models/microscope_model.py", run_name="__main__")

    output_dir = Path("examples/microscope_models/output")
    expected = [
        output_dir / "sample_intensity.png",
        output_dir / "ray_diagram.png",
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Expected output images were not created: {missing}")

    print("Microscope model created and traced.")
    for image in expected:
        print(f"Image: {image}")


if __name__ == "__main__":
    main()
