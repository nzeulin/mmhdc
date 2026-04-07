import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def summarize_completed_runs(values: torch.Tensor):
    """Summarize a `(regularization, dimension, experiment)` tensor with NaN gaps."""
    mean = torch.full(values.shape[:2], torch.nan, dtype=torch.float32)
    p05 = torch.full_like(mean, torch.nan)
    p95 = torch.full_like(mean, torch.nan)

    for reg_idx in range(values.shape[0]):
        for dim_idx in range(values.shape[1]):
            completed = values[reg_idx, dim_idx]
            completed = completed[torch.isfinite(completed)]
            if completed.numel() == 0:
                continue

            mean[reg_idx, dim_idx] = completed.mean()
            p05[reg_idx, dim_idx] = torch.quantile(completed, 0.05)
            p95[reg_idx, dim_idx] = torch.quantile(completed, 0.95)

    return {"mean": mean, "p05": p05, "p95": p95}


def format_regularization_for_filename(regularization: float):
    return f"{regularization:.0e}".replace("+", "")


def get_epoch_curve_path(base_dir: str, dataset_name: str, model_dim: int, regularization: float):
    regularization_token = format_regularization_for_filename(regularization)
    curve_dir = os.path.join(
        base_dir,
        dataset_name,
        f"D={model_dim}",
        f"C={regularization_token}",
    )
    os.makedirs(curve_dir, exist_ok=True)
    return os.path.join(
        curve_dir,
        f"acc_vs_epoch_D={model_dim}_C={regularization_token}.png",
    )


def plot_metric_panel(ax, curves: torch.Tensor, ylabel: str, color: str, title: str):
    epochs = torch.arange(1, curves.shape[1] + 1)
    mean_curve = curves.mean(dim=0)
    p05_curve = torch.quantile(curves, 0.05, dim=0)
    p95_curve = torch.quantile(curves, 0.95, dim=0)

    for run_curve in curves:
        ax.plot(epochs.tolist(), run_curve.tolist(), color=color, alpha=0.15, linewidth=1)

    ax.plot(epochs.tolist(), mean_curve.tolist(), color=color, linewidth=2, label=f"Mean {ylabel.lower()}")
    ax.fill_between(
        epochs.tolist(),
        p05_curve.tolist(),
        p95_curve.tolist(),
        color=color,
        alpha=0.2,
        label="5th-95th percentile",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_epoch_curve(model_dim: int, regularization: float, accuracy_curves: torch.Tensor,
                     loss_curves: torch.Tensor, output_path: str):
    """Update one per-parameter epoch figure with accuracy and loss subplots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_metric_panel(
        axes[0],
        accuracy_curves,
        ylabel="Accuracy",
        color="tab:blue",
        title=f"Accuracy (D={model_dim}, C={regularization:g})",
    )
    plot_metric_panel(
        axes[1],
        loss_curves,
        ylabel="Loss",
        color="tab:orange",
        title=f"Loss (D={model_dim}, C={regularization:g})",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_results(model_dims, regularization_constants, summary, output_path: str):
    """Update the global accuracy-vs-regularization sweep plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")

    mean_accuracies = summary["mean"]
    p05_accuracies = summary["p05"]
    p95_accuracies = summary["p95"]

    finite_mean_accuracies = mean_accuracies[torch.isfinite(mean_accuracies)]
    if finite_mean_accuracies.numel() == 0:
        plt.close(fig)
        return

    min_acc = finite_mean_accuracies.min().item()
    max_acc = finite_mean_accuracies.max().item()

    for dim_idx, model_dim in enumerate(model_dims):
        valid = torch.isfinite(mean_accuracies[:, dim_idx])
        if not valid.any():
            continue

        completed_regularization_constants = [
            regularization_constants[reg_idx]
            for reg_idx, is_valid in enumerate(valid.tolist())
            if is_valid
        ]
        color = cmap(dim_idx % cmap.N)
        ax.plot(
            completed_regularization_constants,
            mean_accuracies[:, dim_idx][valid].tolist(),
            color=color,
            marker="o",
            label=f"D={model_dim:g}",
        )
        ax.fill_between(
            completed_regularization_constants,
            p05_accuracies[:, dim_idx][valid].tolist(),
            p95_accuracies[:, dim_idx][valid].tolist(),
            color=color,
            alpha=0.2,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Regularization constant C")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("MNIST dataset")
    ax.set_ylim(0.85 * min_acc, min(1.0, 1.15 * max_acc))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Hypervector length", ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
