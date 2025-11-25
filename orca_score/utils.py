import os
import json
import yaml

import torch

from torch.utils.tensorboard import SummaryWriter


class CompositeWriter(SummaryWriter):
    """Writer that logs to both TensorBoard and JSONL.

    This class inherits from SummaryWriter and adds JSONL logging capabilities.
    If log_dir is not provided, it will only log to JSONL file without TensorBoard.

    Args:
        log_dir: Directory for TensorBoard logs. If None, TensorBoard logging is disabled.
        jsonl_path: Path to JSONL file for text logging. If None, JSONL logging is disabled.
        **kwargs: Additional arguments passed to SummaryWriter.

    Example:
        >>> writer = CompositeWriter(log_dir="./logs", jsonl_path="./training.jsonl")
        >>> writer.add_scalar("loss", 0.5, step=1)
        >>> writer.add_hparams({}, {"loss": 0.5, "acc": 0.9})
    """

    def __init__(self, log_dir=None, jsonl_path=None, **kwargs):
        self.use_tensorboard = log_dir is not None
        self.jsonl_path = jsonl_path
        if self.use_tensorboard:
            os.makedirs(log_dir, exist_ok=True)
            super().__init__(log_dir=log_dir, **kwargs)
        else:
            # Don't initialize SummaryWriter if no log_dir
            pass

    def add_scalar(self, *args, **kwargs):
        """Add scalar to TensorBoard if enabled."""
        if self.use_tensorboard:
            super().add_scalar(*args, **kwargs)

    def add_hparams(self, hparam_dict, metric_dict, **kwargs):
        """Log hyperparameters and metrics to both TensorBoard and JSONL.

        Args:
            hparam_dict: Dictionary of hyperparameters.
            metric_dict: Dictionary of metrics.
            **kwargs: Additional arguments passed to TensorBoard's add_hparams.
        """
        if self.use_tensorboard:
            # Set run_name to empty string to prevent creating subdirectories
            kwargs.setdefault("run_name", ".")
            super().add_hparams(hparam_dict, metric_dict, **kwargs)

        # Always log to JSONL if path provided
        if self.jsonl_path:
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps({**hparam_dict, **metric_dict}) + "\n")

    def flush(self):
        """Flush TensorBoard writer if enabled."""
        if self.use_tensorboard:
            super().flush()

    def close(self):
        """Close TensorBoard writer if enabled."""
        if self.use_tensorboard:
            super().close()


def save_checkpoint(
    scoring_model,
    output_dir,
    accelerator=None,
    optimizer=None,
    metrics=None,
    args=None,
    tokenizer=None,
):
    """
    Save the model and optimizer state to a checkpoint directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    if accelerator is None:
        scoring_model.save_to_directory(os.path.join(output_dir, "model"))
    else:
        accelerator.unwrap_model(scoring_model).save_to_directory(
            os.path.join(output_dir, "model")
        )
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    if args is not None:
        with open(os.path.join(output_dir, "args.yaml"), "w") as f:
            yaml.dump(vars(args), f)
    if metrics is not None:
        with open(os.path.join(output_dir, "metrics.yaml"), "w") as f:
            yaml.dump(metrics, f)
    if tokenizer is not None:
        tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
