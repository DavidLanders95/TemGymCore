from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")

import jax
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from examples.atoms.multislice_gaussian_benchmark import (
    MultisliceBenchmarkConfig,
    beam_count,
    full_config,
    make_grid_and_beam,
    run_benchmark,
    save_outputs,
)

jax.config.update("jax_enable_x64", True)


def noncentral_diffraction_fraction(fields, center_radius_pixels=1):
    diffraction = np.fft.fftshift(
        np.fft.fft2(fields, axes=(-2, -1)),
        axes=(-2, -1),
    )
    intensity = np.abs(diffraction) ** 2
    cy = intensity.shape[-2] // 2
    cx = intensity.shape[-1] // 2
    central = intensity[
        ...,
        cy - center_radius_pixels : cy + center_radius_pixels + 1,
        cx - center_radius_pixels : cx + center_radius_pixels + 1,
    ]
    total_power = np.sum(intensity, axis=(-2, -1))
    central_power = np.sum(central, axis=(-2, -1))
    return 1.0 - central_power / np.maximum(total_power, 1e-300)


def summarize_result(result, config: MultisliceBenchmarkConfig):
    grid, beam, _, waist = make_grid_and_beam(config)
    del grid
    reference_history = np.asarray(result["reference_history"])
    gaussian_history = np.asarray(result["gaussian_history"])
    reference_diff = noncentral_diffraction_fraction(reference_history)
    gaussian_diff = noncentral_diffraction_fraction(gaussian_history)
    ratio = float(gaussian_diff[-1] / max(reference_diff[-1], 1e-300))

    return {
        "target_beams_side": config.target_beams_side,
        "beam_overlap": config.beam_overlap,
        "fit_samples_per_axis": config.fit_samples_per_axis,
        "fit_support_radius": config.fit_support_radius,
        "beam_count": beam_count(beam),
        "waist_A": float(waist),
        "spacing_A": float((2.0 * result["extent"]) / config.target_beams_side),
        "field_error": float(result["field_error"]),
        "intensity_error": float(result["intensity_error"]),
        "reference_diffracted_power": float(reference_diff[-1]),
        "gaussian_diffracted_power": float(gaussian_diff[-1]),
        "diffracted_power_ratio": ratio,
        "reference_total_s": float(result["timings"]["reference_total"]),
        "gaussian_total_s": float(result["timings"]["gaussian_path_total"]),
        "action_grid_total_s": float(result["timings"]["action_grid_total"]),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    import matplotlib.pyplot as plt

    labels = [
        f"{row['target_beams_side']}x, ov={row['beam_overlap']:g}, "
        f"fit={row['fit_samples_per_axis']}"
        for row in rows
    ]
    x = np.arange(len(rows))
    field_error = np.array([row["field_error"] for row in rows], dtype=float)
    intensity_error = np.array([row["intensity_error"] for row in rows], dtype=float)
    ref_diff = np.array([row["reference_diffracted_power"] for row in rows], dtype=float)
    gauss_diff = np.array([row["gaussian_diffracted_power"] for row in rows], dtype=float)
    gaussian_total = np.array([row["gaussian_total_s"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].plot(x, field_error, marker="o", label="field")
    axes[0].plot(x, intensity_error, marker="o", label="intensity")
    axes[0].set_ylabel("relative error")
    axes[0].set_title("Fresnel comparison")
    axes[0].legend()

    axes[1].plot(x, ref_diff, marker="o", label="Fresnel")
    axes[1].plot(x, gauss_diff, marker="o", label="Gaussian")
    axes[1].set_ylabel("noncentral diffraction power")
    axes[1].set_title("Diffracted power")
    axes[1].legend()

    axes[2].plot(x, gaussian_total, marker="o")
    axes[2].set_ylabel("seconds")
    axes[2].set_title("Gaussian path time")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/atoms/multislice_results/sampling_sweep"))
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--n-slices", type=int, default=4)
    parser.add_argument("--grid-shape", type=int, default=192)
    parser.add_argument("--beam-sides", type=int, nargs="+", default=(224, 320, 448))
    parser.add_argument("--overlaps", type=float, nargs="+", default=(2.0, 1.0))
    parser.add_argument("--fit-samples", type=int, nargs="+", default=(3,))
    parser.add_argument("--fit-support-radius", type=float, default=1.0)
    parser.add_argument("--beam-chunk-size", type=int, default=1792)
    parser.add_argument("--save-each-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"Expected JAX GPU backend, got {jax.default_backend()!r}.")

    base = replace(
        full_config(),
        n_slices=args.n_slices,
        grid_shape=args.grid_shape,
        beam_chunk_size=args.beam_chunk_size,
        fit_support_radius=args.fit_support_radius,
        record_slices=True,
    )

    rows: list[dict[str, object]] = []
    for fit_samples in args.fit_samples:
        for target_beams_side in args.beam_sides:
            for overlap in args.overlaps:
                config = replace(
                    base,
                    target_beams_side=target_beams_side,
                    beam_overlap=overlap,
                    fit_samples_per_axis=fit_samples,
                )
                print(
                    "\n=== "
                    f"side={target_beams_side}, overlap={overlap:g}, "
                    f"fit={fit_samples} ==="
                )
                result = run_benchmark(config)
                row = summarize_result(result, config)
                rows.append(row)
                write_rows(args.output_dir / "sampling_sweep.csv", rows)
                plot_rows(args.output_dir / "sampling_sweep.png", rows)
                if args.save_each_run:
                    run_dir = (
                        args.output_dir
                        / f"side_{target_beams_side}_overlap_{overlap:g}_fit_{fit_samples}"
                    )
                    save_outputs(result, run_dir)

    print("\nSweep results:")
    for row in rows:
        print(
            f"side={row['target_beams_side']}, overlap={row['beam_overlap']}, "
            f"fit={row['fit_samples_per_axis']}, beams={row['beam_count']}, "
            f"waist={row['waist_A']:.4f} A, spacing={row['spacing_A']:.4f} A, "
            f"field={row['field_error']:.4g}, intensity={row['intensity_error']:.4g}, "
            f"diff ratio={row['diffracted_power_ratio']:.3g}, "
            f"gauss time={row['gaussian_total_s']:.3f}s"
        )
    print(f"Saved {args.output_dir / 'sampling_sweep.csv'}")
    print(f"Saved {args.output_dir / 'sampling_sweep.png'}")


if __name__ == "__main__":
    main()
