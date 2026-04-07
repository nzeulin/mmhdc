import os
from pathlib import Path
import sys

from absl import app, flags
from ml_collections.config_flags import config_flags
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import load_mnist
from mmhdc import MultiMMHDC
from mmhdc.utils import HDTransform
from ablation.plot_utils import (
    get_epoch_curve_path,
    plot_epoch_curve,
    plot_results,
    summarize_completed_runs,
)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Path to configuration.")


def evaluate_metrics(model: MultiMMHDC, X_test: torch.Tensor, y_test: torch.Tensor):
    with torch.no_grad():
        y_pred = model(X_test)
        accuracy = (y_pred == y_test).float().mean()
        loss = model.loss(X_test, y_test).detach().to(dtype=torch.float32)
    return accuracy, loss


def load_dataset(config):
    X_train, y_train, X_test, y_test = load_mnist(config.dataset.name)
    return (
        X_train.to(dtype=config.dtype),
        y_train.to(dtype=torch.int64),
        X_test.to(dtype=config.dtype),
        y_test.to(dtype=torch.int64),
    )


def encode_dataset(config, model_dim: int, exp_i: int, X_train: torch.Tensor, X_test: torch.Tensor):
    """Encode one dataset split for a single `(dimension, seed)` pair."""
    transform_dtype = config.model.transform_dtype or config.dtype

    X_train_flat = X_train.view(X_train.shape[0], -1)
    X_test_flat = X_test.view(X_test.shape[0], -1)
    hd_transform = HDTransform(
        in_channels=X_train_flat.shape[1],
        out_channels=model_dim,
        batch_size=config.model.transform_batch_size,
        seed=exp_i,
        normalize=bool(config.model.normalize),
        device=config.device,
        dtype=transform_dtype,
    )
    X_train_hd = hd_transform(X_train_flat)
    X_test_hd = hd_transform(X_test_flat).to(config.device)
    return X_train_hd, X_test_hd


def run_training(config, model_dim: int, regularization: float, exp_i: int,
                 X_train_hd: torch.Tensor, y_train: torch.Tensor,
                 X_test_hd: torch.Tensor, y_test: torch.Tensor):
    """Train one model and return test accuracy/loss after each epoch."""
    model = MultiMMHDC(
        out_channels=model_dim,
        num_classes=config.dataset.num_classes,
        lr=config.model.learning_rate,
        C=regularization,
        device=config.device,
        backend=config.model.backend,
        dtype=config.dtype,
    )
    model.initialize(X_train_hd, y_train)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(exp_i)
    X_train_hd_shuf = X_train_hd.clone()
    y_train_shuf = y_train.clone()
    epoch_accuracies = torch.empty(config.training.num_epochs, dtype=torch.float32)
    epoch_losses = torch.empty(config.training.num_epochs, dtype=torch.float32)
    y_test_device = y_test.to(config.device)

    for epoch in range(config.training.num_epochs):
        if config.training.shuffle:
            idx = torch.randperm(X_train_hd_shuf.shape[0], generator=rng)
            X_train_hd_shuf = X_train_hd_shuf[idx]
            y_train_shuf = y_train_shuf[idx]

        for start in range(0, X_train_hd_shuf.shape[0], config.training.batch_size):
            end = start + config.training.batch_size
            batch_X = X_train_hd_shuf[start:end].to(model.device)
            batch_y = y_train_shuf[start:end].to(model.device)
            model.step(batch_X, batch_y)

        accuracy, loss = evaluate_metrics(model, X_test_hd, y_test_device)
        epoch_accuracies[epoch] = accuracy.cpu()
        epoch_losses[epoch] = loss.cpu()

    return epoch_accuracies, epoch_losses


def save_summary(raw_path: str, config, model_dims, regularization_constants,
                 epoch_accuracies: torch.Tensor, epoch_losses: torch.Tensor,
                 final_accuracies: torch.Tensor,
                 summary, figure_path: str, curves_dir: str):
    """Persist the current ablation state so partial runs remain inspectable."""
    torch.save(
        {
            "config": config.to_dict(),
            "model_dims": model_dims,
            "regularization_constants": regularization_constants,
            "dataset_name": config.dataset.name,
            "epoch_accuracies": epoch_accuracies,
            "epoch_losses": epoch_losses,
            "accuracies": final_accuracies,
            "mean_accuracy": summary["mean"],
            "p05_accuracy": summary["p05"],
            "p95_accuracy": summary["p95"],
            "figure_path": figure_path,
            "curves_dir": curves_dir,
        },
        raw_path,
    )


def main(_):
    torch.manual_seed(0)

    config = FLAGS.config
    if config is None:
        raise ValueError("Configuration file must be provided.")

    os.makedirs(config.paths.results, exist_ok=True)
    curves_dir = os.path.join(config.paths.results, "epoch_curves")
    os.makedirs(curves_dir, exist_ok=True)

    X_train, y_train, X_test, y_test = load_dataset(config)
    model_dims = list(config.sweep.model_dims)
    regularization_constants = list(config.sweep.regularization_constants)
    figure_path = os.path.join(config.paths.results, f"{config.name}.png")
    raw_path = os.path.join(config.paths.results, f"{config.name}.pt")

    epoch_accuracies = torch.full(
        (
            len(regularization_constants),
            len(model_dims),
            config.training.num_experiments,
            config.training.num_epochs,
        ),
        torch.nan,
        dtype=torch.float32,
    )
    epoch_losses = torch.full(
        (
            len(regularization_constants),
            len(model_dims),
            config.training.num_experiments,
            config.training.num_epochs,
        ),
        torch.nan,
        dtype=torch.float32,
    )
    final_accuracies = torch.full(
        (
            len(regularization_constants),
            len(model_dims),
            config.training.num_experiments,
        ),
        torch.nan,
        dtype=torch.float32,
    )

    total_runs = len(model_dims) * len(regularization_constants) * config.training.num_experiments
    progress = tqdm(total=total_runs, desc="Ablation runs", unit="run")

    # Reuse the expensive HD encoding across all regularization values for a fixed
    # `(dimension, seed)` pair. Only the classifier training is repeated per `C`.
    for exp_i in range(config.training.num_experiments):
        for dim_idx, model_dim in enumerate(model_dims):
            X_train_hd, X_test_hd = encode_dataset(config, model_dim, exp_i, X_train, X_test)
            for reg_idx, regularization in enumerate(regularization_constants):
                run_epoch_accuracies, run_epoch_losses = run_training(
                    config,
                    model_dim,
                    float(regularization),
                    exp_i,
                    X_train_hd,
                    y_train,
                    X_test_hd,
                    y_test,
                )
                epoch_accuracies[reg_idx, dim_idx, exp_i] = run_epoch_accuracies
                epoch_losses[reg_idx, dim_idx, exp_i] = run_epoch_losses
                final_accuracy = run_epoch_accuracies[-1]
                final_accuracies[reg_idx, dim_idx, exp_i] = final_accuracy

                completed_accuracy_curves = epoch_accuracies[reg_idx, dim_idx, :exp_i + 1]
                completed_loss_curves = epoch_losses[reg_idx, dim_idx, :exp_i + 1]
                curve_path = get_epoch_curve_path(
                    curves_dir,
                    config.dataset.name,
                    model_dim,
                    float(regularization),
                )
                plot_epoch_curve(
                    model_dim,
                    float(regularization),
                    completed_accuracy_curves,
                    completed_loss_curves,
                    curve_path,
                )

                progress.set_postfix(
                    exp=f"{exp_i + 1}/{config.training.num_experiments}",
                    C=f"{regularization:g}",
                    D=model_dim,
                    acc=f"{final_accuracy.item():.4f}",
                )
                progress.update(1)

                # Refresh the global sweep plot after every completed parameter-set run.
                summary = summarize_completed_runs(final_accuracies)
                plot_results(model_dims, regularization_constants, summary, figure_path)

                # Persist the current summary after every completed parameter-set run.
                save_summary(
                    raw_path,
                    config,
                    model_dims,
                    regularization_constants,
                    epoch_accuracies,
                    epoch_losses,
                    final_accuracies,
                    summary,
                    figure_path,
                    curves_dir,
                )

    progress.close()

    summary = summarize_completed_runs(final_accuracies)

    plot_results(model_dims, regularization_constants, summary, figure_path)
    save_summary(
        raw_path,
        config,
        model_dims,
        regularization_constants,
        epoch_accuracies,
        epoch_losses,
        final_accuracies,
        summary,
        figure_path,
        curves_dir,
    )

    print(f"Saved raw ablation results to {raw_path}")
    print(f"Saved plot to {figure_path}")


if __name__ == "__main__":
    app.run(main)
