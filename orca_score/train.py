"""Authors: Bolaji Yusuf, Santosh Kesiraju"""

import argparse
import os
import platform
import random
import sys
from datetime import datetime
from functools import partial

import accelerate
import numpy as np
import torch
import tqdm
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from orca_score import data, model
from orca_score.utils import CompositeWriter, save_checkpoint


def write_to_log(args, total_params, trainable_params, total_batch_size, writer, accelerator):
    """Write model and training configuration to log file."""

    writer.log("== MODEL CONFIGURATION ==")
    writer.log(f"Model: {args.model}")
    writer.log(f"Score type: {args.score_type}")
    writer.log(f"Quantization: {args.quantization_level}")
    writer.log(f"LoRA rank: {args.lora_rank if args.lora_rank else 'None (full fine-tuning)'}")
    writer.log(f"Use CLS token: {args.use_cls_token}")
    writer.log(f"Flash Attention 2: {args.use_flash_attention}")
    writer.log(f"Total parameters: {total_params:.1f}M", to_console=True)
    writer.log(f"Trainable parameters: {trainable_params:.1f}M", to_console=True)
    writer.log(f"Trainable ratio: {trainable_params/total_params*100:.2f}%", to_console=True)

    writer.log("== TRAINING CONFIGURATION ==")
    writer.log(f"Total batch size: {total_batch_size}")
    writer.log(f"  - Processes: {accelerator.num_processes}")
    writer.log(f"  - Per-device batch size: {args.batch_size}")
    writer.log(f"  - Gradient accumulation steps: {args.accumulation_steps}")
    writer.log(f"Max steps: {args.max_steps}")
    writer.log(f"Validation interval: {args.val_steps} steps")
    writer.log(f"Save interval: {args.save_steps} steps")
    writer.log(f"Peak learning rate: {args.peak_lr}")
    writer.log(f"Min LR ratio: {args.min_lr_ratio}")
    writer.log(f"Warmup steps: {args.warmup_steps}")
    writer.log(f"Weight decay: {args.weight_decay}")
    writer.log(f"Max gradient norm: {args.max_grad_norm}")
    writer.log(f"LR ratio classifier: {args.lr_ratio_classifier}")
    writer.log(f"Early stopping patience: {args.early_stopping_patience}")

    writer.log("== DATA CONFIGURATION ==")
    writer.log(f"Training data: {args.train_data}")
    writer.log(f"Validation data: {args.val_data}")
    writer.log(f"Max data length: {args.max_data_length}")
    writer.log(f"Skip rationale: {args.skip_rationale}")
    writer.log(f"Skip question: {args.skip_question}")
    writer.log(f"Add transcript: {args.add_transcript}")


def set_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_lambda_linear_with_min_lr(step, args):
    """Linear learning rate scheduler with minimum learning rate ratio."""
    peak_lr = args.peak_lr
    min_lr = peak_lr * args.min_lr_ratio
    if step < args.warmup_steps:
        return step / args.warmup_steps
    slope = (peak_lr - min_lr) / (args.max_steps - args.warmup_steps)
    intercept = peak_lr - slope * args.warmup_steps
    current_lr = -slope * (step - args.warmup_steps) + intercept
    return max(current_lr, min_lr) / peak_lr


def parse_arguments():
    """Parse command line arguments for training the ORCA model."""

    parser = argparse.ArgumentParser(
        description="Train ORCA model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_group = parser.add_argument_group("Input data related arguments")

    data_group.add_argument(
        "--train_data",
        type=str,
        nargs="+",
        required=True,
        help="Paths to training data files. Can be multiple files for different splits.",
    )
    data_group.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="Path to validation data file.",
    )
    data_group.add_argument(
        "--max_data_length",
        type=int,
        default=1000,
        help="Maximum length of input text in characters.",
    )
    data_group.add_argument(
        "--dataset_sampling_weights",
        type=float,
        nargs="+",
        default=None,
        help="Weights for sampling from different datasets. "
        "If not provided, datasets will just be concatenated.",
    )
    data_group.add_argument(
        "--skip_rationale",
        action="store_true",
        help="If set, the model will not use rationales for scoring.",
    )
    data_group.add_argument(
        "--skip_question",
        action="store_true",
        help="If set, the model will not use questions for scoring.",
    )
    data_group.add_argument(
        "--add_transcript",
        action="store_true",
        help="If set, the model will use transcriptions for scoring.",
    )

    data_group.add_argument(
        "--prompts_yaml",
        type=str,
        default=None,
        help="Path to YAML file with prompts for the model.",
    )

    model_group = parser.add_argument_group("Model related arguments")

    # Core architecture
    model_group.add_argument(
        "--model",
        type=str,
        default="allenai/OLMo-2-0425-1B-Instruct",
        help="LLM to initialize from.",
    )
    model_group.add_argument(
        "--tokenizer",
        type=str,
        help="Path to the tokenizer directory. Defaults to model if unset",
    )
    model_group.add_argument(
        "--score_type",
        type=str,
        default="beta",
        choices=["bernoulli", "beta"],
        help="Type of scoring to use for training the model.",
    )
    model_group.add_argument(
        "--use_cls_token",
        action="store_true",
        help="If set, the model will append a CLS token for scoring.",
    )

    # Optimization and efficiency
    model_group.add_argument("--lora_rank", type=int, help="LoRA rank, default is no LoRA")
    model_group.add_argument(
        "--quantization_level",
        type=str,
        default="none",
        choices=["none", "4bit", "8bit"],
        help="Quantization level to use for the model.",
    )
    model_group.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="If set, the model will use Flash Attention 2 for faster attention computation. Requires flash-attn package.",
    )

    # Initialization
    model_group.add_argument(
        "--init_type",
        type=str,
        default="xavier_normal",
        choices=["xavier_normal", "kaiming_normal"],
        help="Initialization method for the linear scoring layer that outputs log(alpha), log(beta).",
    )

    training_group = parser.add_argument_group("Training related arguments")

    # Checkpointing and resume
    training_group.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to load the model from.",
    )
    training_group.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume training from the latest checkpoint.",
    )

    # I/O paths
    training_group.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        required=True,
        help="Directory to save the trained model.",
    )
    training_group.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        required=True,
        help="Directory to save logs for TensorBoard.",
    )
    training_group.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name for logging. If not provided, will be auto-generated from hyperparameters.",
    )

    # Training schedule
    training_group.add_argument(
        "--max_steps", type=int, default=4000, help="Number of training steps."
    )
    training_group.add_argument(
        "--val_steps", type=int, default=200, help="Number of steps between validation."
    )
    training_group.add_argument(
        "--save_steps", type=int, default=500, help="Number of steps between saves."
    )
    training_group.add_argument(
        "--log_steps", type=int, default=50, help="Number of steps between logging training loss."
    )
    training_group.add_argument(
        "--early_stopping_patience",
        type=int,
        default=30,
        help="Number of validation steps without improvement before early stopping.",
    )

    # Batch size and accumulation
    training_group.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training."
    )
    training_group.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )

    # Optimization hyperparameters
    training_group.add_argument(
        "--peak_lr", type=float, default=5e-5, help="Peak learning rate for training."
    )
    training_group.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.01,
        help="Minimum learning rate ratio relative to peak learning rate.",
    )
    training_group.add_argument(
        "--lr_ratio_classifier",
        type=float,
        default=1.0,
        help="Learning rate ratio for the classifier head relative to the LM.",
    )
    training_group.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps for learning rate.",
    )
    training_group.add_argument(
        "--weight_decay", type=float, default=0, help="Weight decay for optimizer."
    )
    training_group.add_argument(
        "--max_grad_norm",
        type=float,
        default=5.0,
        help="Maximum gradient norm for clipping.",
    )

    # Miscellaneous
    training_group.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for data loading."
    )
    training_group.add_argument(
        "--seed",
        type=int,
        default=108,
        help="Random seed for reproducibility.",
    )
    training_group.add_argument(
        "--verbose",
        action="store_true",
        help="If set, the training process will output more detailed logs.",
    )

    args = parser.parse_args()

    return args


def get_dataloaders(args, tokenizer, writer):
    """Create training and validation dataloaders."""

    if args.prompts_yaml is not None:
        prompts = yaml.safe_load(open(args.prompts_yaml, "r"))
    else:
        prompts = None

    collate_fn = data.CollateFn(tokenizer)
    if args.dataset_sampling_weights is None:
        train_dataset = data.ConcatenatedDataset(
            [
                data.UnifiedAnnotationDataset(
                    jfile,
                    prompts=prompts,
                    filter_func=lambda x: len(x["text"]) <= args.max_data_length,
                    skip_rationale=args.skip_rationale,
                    skip_question=args.skip_question,
                    add_transcript=args.add_transcript,
                    log_fn=lambda msg: writer.log(msg, to_console=True),
                )
                for jfile in args.train_data
            ]
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
        train_datasets = [
            data.UnifiedAnnotationDataset(
                jfile,
                prompts=prompts,
                filter_func=lambda x: len(x["text"]) <= args.max_data_length,
                skip_rationale=args.skip_rationale,
                skip_question=args.skip_question,
                add_transcript=args.add_transcript,
                log_fn=lambda msg: writer.log(msg, to_console=True),
            )
            for jfile in args.train_data
        ]

        train_dataloaders = [
            DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_fn,
                pin_memory=True,
            )
            for train_dataset in train_datasets
        ]
        train_dataset = data.DatasetWithSampling(
            train_dataloaders, sampling_weights=args.dataset_sampling_weights
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda x: x[0],
        )

    val_dataset = data.UnifiedAnnotationDataset(
        args.val_data,
        prompts=prompts,
        skip_rationale=args.skip_rationale,
        skip_question=args.skip_question,
        log_fn=lambda msg: writer.log(msg, to_console=True),
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    writer.log(f"Total number of training samples: {len(train_dataset)}", to_console=True)
    writer.log(f"Total number of validation samples: {len(val_dataset)}", to_console=True)

    return train_dataloader, val_dataloader


def build_model(args, accelerator, device, writer):
    """Build the ORCA model and tokenizer, loading from checkpoint if specified."""

    # Load the model and tokenizer
    attn_impl = "flash_attention_2" if args.use_flash_attention else None

    # Log model loading info
    if accelerator.is_main_process:
        precision = args.quantization_level if args.quantization_level != "none" else "bfloat16"
        flash_attn = " with Flash Attention 2" if args.use_flash_attention else ""
        writer.log(f"Loading {args.model} with {precision} precision{flash_attn}", to_console=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.quantization_level == "4bit",
        load_in_8bit=args.quantization_level == "8bit",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    if args.quantization_level == "none":
        lm = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map=device.type,
            low_cpu_mem_usage=True,
        )
    else:
        lm = AutoModelForCausalLM.from_pretrained(
            args.model,
            attn_implementation=attn_impl,
            quantization_config=bnb_config,
            device_map=device.type,
            low_cpu_mem_usage=True,
        )

    if args.tokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    if args.lora_rank is not None and not args.load_checkpoint:
        # Exclude lm_head from LoRA since we don't use it for scoring
        # Target attention and MLP layers but not lm_head
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
        )
        lm = get_peft_model(lm, lora_config)

        # Safety check: If embeddings are tied with lm_head, we need to be careful
        # Most models (Gemma, Llama) don't tie embeddings, but check anyway
        if hasattr(lm.config, "tie_word_embeddings") and lm.config.tie_word_embeddings:
            import ipdb

            ipdb.set_trace()
            print(lm.config)
            if accelerator.is_main_process:
                writer.log(
                    "WARNING: Model has tied word embeddings. Input embeddings share weights with lm_head.",
                    level="WARNING",
                    to_console=True,
                )

    scoring_model = model.ORCA(
        lm,
        score_type=args.score_type,
        use_cls_token=args.use_cls_token,
        init_type=args.init_type,
    ).to(torch.bfloat16)

    # Freeze lm_head if doing full fine-tuning (not using it for scoring)
    # Skip if using LoRA (already excluded from target_modules)
    # Skip if embeddings are tied (freezing lm_head would freeze embeddings too)
    if args.lora_rank is None:
        embeddings_tied = getattr(lm.config, "tie_word_embeddings", False)
        if hasattr(lm, "lm_head") and not embeddings_tied:
            for param in lm.lm_head.parameters():
                param.requires_grad = False
            if accelerator.is_main_process:
                writer.log("Frozen lm_head (not used for scoring)", to_console=True)
        elif hasattr(lm, "lm_head") and embeddings_tied:
            if accelerator.is_main_process:
                writer.log(
                    "WARNING: Keeping lm_head trainable because embeddings are tied",
                    level="WARNING",
                    to_console=True,
                )

    # Load existing checkpoint if specified

    start_step = 0
    best_val_rho = float("-inf")

    if args.load_checkpoint:
        if accelerator.is_main_process:
            writer.log(f"Loading model from checkpoint: {args.load_checkpoint}", to_console=True)
        if args.lora_rank is not None:
            lm = PeftModel.from_pretrained(lm, os.path.join(args.load_checkpoint, "lm"))
            if accelerator.is_main_process:
                writer.log("Loaded LoRA model from checkpoint", to_console=True)
        else:
            lm = AutoModel.from_pretrained(
                os.path.join(args.load_checkpoint, "lm"),
                dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )
            if accelerator.is_main_process:
                writer.log("Loaded LM model from checkpoint", to_console=True)
        scoring_model = model.ORCA.load_from_directory(args.load_checkpoint, lm, device=device)
    if args.resume:
        if accelerator.is_main_process:
            writer.log("Resuming from latest checkpoint", to_console=True)
        latest_checkpoint = os.path.join(args.output_dir, "latest")
        if args.lora_rank is not None:
            lm = PeftModel.from_pretrained(lm, os.path.join(latest_checkpoint, "lm"))
            if accelerator.is_main_process:
                writer.log("Loaded LoRA model from checkpoint", to_console=True)
        else:
            lm = AutoModel.from_pretrained(
                os.path.join(latest_checkpoint, "lm"),
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )
            if accelerator.is_main_process:
                writer.log("Loaded LM model from checkpoint", to_console=True)
        scoring_model = model.ORCA.load_from_directory(latest_checkpoint, lm, device=device)
        with open(os.path.join(latest_checkpoint, "metrics.yaml"), "r") as f:
            all_metrics = yaml.safe_load(f)
        start_step = all_metrics[-1]["step"] if all_metrics else 0
        best_val_rho = (
            all_metrics[-1].get("mean_rho", best_val_rho) if all_metrics else float("-inf")
        )

    return scoring_model, tokenizer, start_step, best_val_rho


def build_optimizer_and_scheduler(args, scoring_model, accelerator, writer):

    optimizer_dict_list = []
    optimizer_dict_list.append(
        {
            "params": [p for p in scoring_model.lm.parameters() if p.requires_grad],
            "lr": args.peak_lr,
        }
    )
    optimizer_dict_list.append(
        {
            "params": scoring_model.linear.parameters(),
            "lr": args.peak_lr * args.lr_ratio_classifier,
        }
    )
    optimizer = torch.optim.AdamW(
        optimizer_dict_list, lr=args.peak_lr, weight_decay=args.weight_decay
    )
    if args.resume:
        if accelerator.is_main_process:
            writer.log("Resuming optimizer state from checkpoint", to_console=True)
        latest_checkpoint = os.path.join(args.output_dir, "latest")
        optimizer.load_state_dict(torch.load(os.path.join(latest_checkpoint, "optimizer.pt")))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=partial(lr_lambda_linear_with_min_lr, args=args)
    )
    return optimizer, lr_scheduler


@torch.no_grad()
def evaluate(
    scoring_model,
    val_dataloader,
    accelerator,
    logging_dict,
    step,
    writer,
    best_val_rho,
    args,
):
    """Evaluate the model on the validation set and log metrics."""

    device = accelerator.device

    val_loss = 0
    val_num_samples = 0
    all_val_annotations = []
    all_val_predictions = []
    all_val_params = []

    all_val_variance_annotations = []
    all_val_variance_predictions = []

    for val_batch in tqdm.tqdm(val_dataloader, disable=not accelerator.is_main_process):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            val_outputs = scoring_model(val_batch)
            loss = val_outputs["loss"]  # This is sum of losses in the batch

        batch_size = val_batch["input_ids"].shape[0]

        # Gather loss sum and batch size across all processes
        gathered_loss_sum = accelerator.gather(loss.detach()).sum().item()
        gathered_batch_size = (
            accelerator.gather(torch.tensor(batch_size, device=loss.device)).sum().item()
        )

        val_loss += gathered_loss_sum
        val_num_samples += gathered_batch_size

        val_annotation = val_batch["average_labels"]
        val_prediction = val_outputs["prob1"]
        all_val_annotations.extend(val_annotation.cpu().tolist())
        all_val_predictions.extend(val_prediction.cpu().tolist())
        all_val_params.extend(val_outputs["params"].cpu().tolist())

        if "label_variance" in val_batch:
            val_variance_annotation = val_batch["label_variance"]
            val_variance_prediction = val_outputs["variance"]
            all_val_variance_annotations.extend(val_variance_annotation.cpu().tolist())
            all_val_variance_predictions.extend(val_variance_prediction.cpu().tolist())
    metrics_to_sync = torch.tensor([val_loss, val_num_samples], device=device)
    metrics_to_sync = accelerator.gather(metrics_to_sync)
    val_loss = metrics_to_sync[0].sum().item()
    val_num_samples = metrics_to_sync[1].sum().item()

    with open(
        os.path.join(
            args.output_dir,
            f"val_predictions_{step}_{accelerator.process_index}.yaml",
        ),
        "w",
    ) as f:
        yaml.dump(
            {
                "predictions": all_val_predictions,
                "annotations": all_val_annotations,
                "params": all_val_params,
                "predictions_variance": all_val_variance_predictions,
                "annotations_variance": all_val_variance_annotations,
            },
            f,
        )
        accelerator.wait_for_everyone()

    all_val_annotations = []
    all_val_predictions = []
    all_val_params = []
    all_val_variance_annotations = []
    all_val_variance_predictions = []
    for i in range(accelerator.num_processes):
        with open(os.path.join(args.output_dir, f"val_predictions_{step}_{i}.yaml"), "r") as f:
            result_data = yaml.safe_load(f)
            all_val_annotations.extend(result_data["annotations"])
            all_val_predictions.extend(result_data["predictions"])
            all_val_params.extend(result_data["params"])
            all_val_variance_annotations.extend(result_data["annotations_variance"])
            all_val_variance_predictions.extend(result_data["predictions_variance"])

    # Calculate Kendall's tau and Spearman's rank correlation
    kendall_tau = kendalltau(all_val_annotations, all_val_predictions)
    spearman_corr = spearmanr(all_val_annotations, all_val_predictions)

    variance_kendall_tau = kendalltau(all_val_variance_annotations, all_val_variance_predictions)
    variance_spearman_corr = spearmanr(all_val_variance_annotations, all_val_variance_predictions)

    if accelerator.is_main_process:
        writer.add_scalar("Loss/val_loss_mean", val_loss / val_num_samples, step)
        writer.add_scalar(f"Loss/val_loss_mean_{args.score_type}", val_loss / val_num_samples, step)
        writer.add_scalar("Correlations/Mean Kendall's Tau", float(kendall_tau.statistic), step)
        writer.add_scalar(
            "Correlations/Mean Spearman's Rank Correlation",
            float(spearman_corr.statistic),
            step,
        )
        writer.add_scalar(
            "Correlations/Variance Kendall's Tau",
            float(variance_kendall_tau.statistic),
            step,
        )
        writer.add_scalar(
            "Correlations/Variance Spearman's Rank Correlation",
            float(variance_spearman_corr.statistic),
            step,
        )

    logging_dict.update(
        {
            "val_loss": round(val_loss / val_num_samples, 6),
            "best_val_rho": round(max(best_val_rho, float(spearman_corr.statistic)), 6),
            "variance_tau": round(float(variance_kendall_tau.statistic), 6),
            "variance_rho": round(float(variance_spearman_corr.statistic), 6),
            "mean_tau": round(float(kendall_tau.statistic), 6),
            "mean_rho": round(float(spearman_corr.statistic), 6),
        }
    )
    if accelerator.is_main_process:
        writer.log(str(logging_dict), to_console=True)

    # Log to JSONL (and TensorBoard if enabled)
    if accelerator.is_main_process:
        writer.add_hparams({}, logging_dict)

    return spearman_corr, logging_dict


def main():

    args = parse_arguments()

    # Set random seeds for reproducibility
    set_seeds(args.seed)

    ddp_kwargs = accelerate.utils.DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    torch.tensor(0).to(accelerator.device)
    device = accelerator.device

    # Generate or use provided experiment name
    if not args.exp_name:
        model_name = args.model.split("/")[-1]
        effective_batch = args.batch_size * args.accumulation_steps * accelerator.num_processes
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = (
            f"{model_name}_{args.score_type}_lr{args.peak_lr}_bs{effective_batch}__{timestamp}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    args_dict = {arg: getattr(args, arg) for arg in vars(args)}
    if not os.path.exists(os.path.join(args.output_dir, "args.yaml")):
        with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
            yaml.dump(args_dict, f)

    # Create writer for logging
    jsonl_log_path = os.path.join(args.output_dir, "training_metrics.jsonl")
    tensorboard_dir = os.path.join(args.log_dir, args.exp_name) if args.log_dir else None
    writer = CompositeWriter(
        log_dir=tensorboard_dir,
        jsonl_path=jsonl_log_path,
        resume=args.resume,
        console_print_fn=accelerator.print,
    )

    writer.log(f"Platform: {platform.node()} {platform.platform()}", to_console=True)
    writer.log(f"Experiment name: {args.exp_name}", to_console=True)
    writer.log(f"TensorBoard logs: {tensorboard_dir}", to_console=True)
    writer.log(f"Training log: {writer.text_log_path}", to_console=True)

    # Log initial configuration (only main process writes)
    if accelerator.is_main_process:
        writer.log(" ".join(sys.argv))

    # Build model and tokenizer - load from checkpoint if specified
    scoring_model, tokenizer, start_step, best_val_rho = build_model(
        args, accelerator, device, writer
    )
    writer.log(scoring_model)

    # Create datasets and dataloaders
    train_dataloader, val_dataloader = get_dataloaders(args, tokenizer, writer)

    optimizer, lr_scheduler = build_optimizer_and_scheduler(
        args, scoring_model, accelerator, writer
    )

    scoring_model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        scoring_model, optimizer, train_dataloader, val_dataloader
    )

    total_params = sum(p.numel() / 1e6 for p in scoring_model.parameters())
    trainable_params = sum(p.numel() / 1e6 for p in scoring_model.parameters() if p.requires_grad)
    total_batch_size = args.batch_size * accelerator.num_processes * args.accumulation_steps

    # Log model and training configuration to file
    if accelerator.is_main_process:
        write_to_log(args, total_params, trainable_params, total_batch_size, writer, accelerator)

        if args.resume:
            writer.log(f"RESUMING FROM CHECKPOINT (step {start_step})", to_console=True)
            writer.log(f"Best validation rho so far: {best_val_rho:.6f}", to_console=True)

    all_metrics = []
    metrics_at_best_checkpoint = []
    train_loss = 0
    train_count = 0

    train_loader_iter = iter(train_dataloader)
    scoring_model.train()

    for j in tqdm.tqdm(
        range(start_step + 1, args.max_steps + 1),
        disable=not args.verbose and not accelerator.is_main_process,
    ):
        step_loss_sum = 0  # Sum of losses across all samples
        step_num_samples = 0  # Total number of samples
        for k in range(args.accumulation_steps):
            with accelerator.accumulate(scoring_model):
                try:
                    batch = next(train_loader_iter)
                except StopIteration:
                    train_loader_iter = iter(train_dataloader)
                    batch = next(train_loader_iter)

                batch_size = batch["input_ids"].shape[0]

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    outputs = scoring_model(batch)
                    loss = outputs["loss"]  # This is now sum of losses in the batch

                # Check for abnormally high loss and log diagnostics
                if loss.item() > 1000 or torch.isnan(loss) or torch.isinf(loss):
                    if accelerator.is_main_process:
                        writer.log(
                            f"WARNING: High/invalid loss detected at step {j}, accumulation {k+1}!",
                            level="WARNING",
                        )
                        writer.log(f"  Loss: {loss.item()}", level="WARNING")
                        if "params" in outputs:
                            params = outputs["params"]
                            concentrations = params.exp()
                            writer.log(
                                f"  Concentration1 range: [{concentrations[:, 0].min().item():.4f}, {concentrations[:, 0].max().item():.4f}]",
                                level="WARNING",
                            )
                            writer.log(
                                f"  Concentration0 range: [{concentrations[:, 1].min().item():.4f}, {concentrations[:, 1].max().item():.4f}]",
                                level="WARNING",
                            )
                        if "labels" in batch:
                            valid_labels = batch["labels"][batch["labels"] >= 0]
                            if len(valid_labels) > 0:
                                writer.log(
                                    f"  Label range: [{valid_labels.min().item():.4f}, {valid_labels.max().item():.4f}]",
                                    level="WARNING",
                                )

                accelerator.backward(loss)

                # Gather the sum of losses across all processes
                gathered_loss_sum = accelerator.gather(loss.detach()).sum().item()
                # Gather total batch size across all processes
                gathered_batch_size = (
                    accelerator.gather(torch.tensor(batch_size, device=loss.device)).sum().item()
                )

                step_loss_sum += gathered_loss_sum
                step_num_samples += gathered_batch_size

                if args.verbose:
                    accelerator.print(
                        f"Step {j}, Accumulation {k+1}/{args.accumulation_steps}, "
                        f"Batch Loss Sum: {gathered_loss_sum:.4f}, Batch Size: {gathered_batch_size}, "
                        f"Batch Avg: {gathered_loss_sum/gathered_batch_size:.4f}, Sync: {accelerator.sync_gradients}"
                    )

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        scoring_model.parameters(), args.max_grad_norm
                    )
                    # Average loss across all samples in this step
                    avg_step_loss = step_loss_sum / step_num_samples
                    train_loss += avg_step_loss
                    train_count += 1
                    if args.verbose:
                        accelerator.print(
                            f"Step {j}: Step Loss Sum: {step_loss_sum:.4f}, Num Samples: {step_num_samples}, "
                            f"Avg Loss: {avg_step_loss:.4f}, Running Avg: {train_loss/train_count:.4f}, Grad Norm: {grad_norm:.4f}"
                        )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if j % args.log_steps == 0 and accelerator.is_main_process and train_count > 0:
            writer.log(
                f"Step {j}: Training loss: {train_loss/train_count:.4f} Learning Rate: {optimizer.param_groups[0]['lr']:.6e}",
                to_console=True,
            )

        if accelerator.is_main_process and train_count > 0:
            writer.add_scalar("Loss/train_loss", train_loss / train_count, j)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], j)
            writer.add_scalar("Data/Text Length", batch["input_len"].float().mean().item(), j)
            writer.add_scalar("Data/Batch Size", batch["input_ids"].shape[0], j)
            if accelerator.sync_gradients:
                writer.add_scalar("Loss/Grad Norm", grad_norm, j)

        if j % args.val_steps == 0:

            scoring_model.eval()

            logging_dict = {
                "step": j,
                "train_loss": round(train_loss / train_count, 6),
                "lr": optimizer.param_groups[0]["lr"],
            }

            spearman_corr, logging_dict = evaluate(
                scoring_model,
                val_dataloader,
                accelerator,
                logging_dict,
                j,
                writer,
                best_val_rho,
                args,
            )
            all_metrics.append(logging_dict)
            if float(spearman_corr.statistic) > best_val_rho:
                best_val_rho = float(spearman_corr.statistic)
                save_checkpoint(
                    scoring_model,
                    os.path.join(args.output_dir, "best"),
                    accelerator=accelerator,
                    metrics=all_metrics,
                    args=args,
                    tokenizer=tokenizer,
                )
                metrics_at_best_checkpoint = logging_dict
                if accelerator.is_main_process:
                    writer.log(
                        f"Step {j}: New best validation rho: {best_val_rho:.6f} - Saved checkpoint to 'best'",
                        to_console=True,
                    )
            if accelerator.is_main_process:
                save_checkpoint(
                    scoring_model,
                    os.path.join(args.output_dir, "latest"),
                    accelerator=accelerator,
                    metrics=all_metrics,
                    args=args,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                )

        if accelerator.is_main_process and j % args.save_steps == 0:
            save_checkpoint(
                scoring_model,
                os.path.join(args.output_dir, f"checkpoint_{j}"),
                accelerator=accelerator,
                metrics=all_metrics,
                args=args,
                tokenizer=tokenizer,
            )
            writer.log(f"Step {j}: Saved periodic checkpoint to 'checkpoint_{j}'")

        scoring_model.train()

        recent_val_rho = [m["mean_rho"] for m in all_metrics[-args.early_stopping_patience :]]
        all_val_rhos = [m["mean_rho"] for m in all_metrics]
        if len(all_val_rhos) > args.early_stopping_patience and all(
            v < max(all_val_rhos) for v in recent_val_rho
        ):
            if accelerator.is_main_process:
                writer.log(f"EARLY STOPPING at step {j}", to_console=True)
                writer.log(
                    f"No improvement in validation rho for {args.early_stopping_patience} validation steps",
                    to_console=True,
                )
                writer.log(f"Best validation rho: {max(all_val_rhos):.6f}", to_console=True)

            break

    if accelerator.is_main_process:
        writer.log("== TRAINING COMPLETED ==", to_console=True)
        writer.log(
            f"Training finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", to_console=True
        )
        writer.log(f"Total steps: {j}", to_console=True)
        if metrics_at_best_checkpoint:
            writer.log("\nBest checkpoint metrics:", to_console=True)
            for key, value in metrics_at_best_checkpoint.items():
                writer.log(f"  {key}: {value}", to_console=True)
        writer.close()


if __name__ == "__main__":
    main()
