import argparse
import os
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ablation.plot_utils import get_epoch_curve_path, plot_epoch_curve, plot_results, summarize_completed_runs


def get_completed_epoch_curves(curves: torch.Tensor):
    """Return only completed experiment curves from a `(experiment, epoch)` tensor."""
    completed_mask = torch.isfinite(curves).all(dim=1)
    return curves[completed_mask]


def resolve_output_paths(payload: dict, input_path: str, output_dir: str | None):
    if output_dir is None:
        figure_path = payload["figure_path"]
        curves_dir = payload["curves_dir"]
        return figure_path, curves_dir

    os.makedirs(output_dir, exist_ok=True)
    figure_name = Path(payload.get("figure_path", Path(input_path).with_suffix(".png").name)).name
    figure_path = os.path.join(output_dir, figure_name)
    curves_dir = os.path.join(output_dir, "epoch_curves")
    os.makedirs(curves_dir, exist_ok=True)
    return figure_path, curves_dir


def main():
    parser = argparse.ArgumentParser(description="Rebuild ablation plots from a saved .pt file.")
    parser.add_argument("results_path", help="Path to the saved ablation .pt file.")
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to the paths stored in the .pt file.",
    )
    args = parser.parse_args()

    payload = torch.load(args.results_path, map_location="cpu")

    model_dims = payload["model_dims"]
    regularization_constants = payload["regularization_constants"]
    final_accuracies = payload["accuracies"]
    epoch_accuracies = payload["epoch_accuracies"]
    epoch_losses = payload["epoch_losses"]
    dataset_name = payload.get("dataset_name", payload["config"]["dataset"]["name"])

    figure_path, curves_dir = resolve_output_paths(payload, args.results_path, args.output_dir)

    summary = summarize_completed_runs(final_accuracies)
    plot_results(model_dims, regularization_constants, summary, figure_path)

    for dim_idx, model_dim in enumerate(model_dims):
        for reg_idx, regularization in enumerate(regularization_constants):
            accuracy_curves = get_completed_epoch_curves(epoch_accuracies[reg_idx, dim_idx])
            loss_curves = get_completed_epoch_curves(epoch_losses[reg_idx, dim_idx])
            if accuracy_curves.numel() == 0 or loss_curves.numel() == 0:
                continue

            curve_path = get_epoch_curve_path(curves_dir, dataset_name, model_dim, float(regularization))
            plot_epoch_curve(
                model_dim,
                float(regularization),
                accuracy_curves,
                loss_curves,
                curve_path,
            )

    print(f"Rebuilt global plot at {figure_path}")
    print(f"Rebuilt epoch plots under {curves_dir}")


if __name__ == "__main__":
    main()
