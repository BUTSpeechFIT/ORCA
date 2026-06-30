"""Author: Bolaji Yusuf, Santosh Kesiraju"""

import argparse
import json
import os

import torch
import tqdm
import yaml
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from orca_score import data, model, utils


def _is_labeled(jsonl_path: str) -> bool:
    """Return True if the first row of the JSONL has at least one valid rating."""
    with open(jsonl_path, encoding="utf-8") as f:
        first_line = f.readline()
    row = json.loads(first_line)
    return any(r is not None and 0 <= r <= 5 for r in row.get("ratings", []))


def main():
    """Main function for ORCA inference."""

    args = parse_arguments()

    if not args.output_dir:
        args.output_dir = os.path.join(os.path.dirname(args.model_path), "inference_results")
    os.makedirs(args.output_dir, exist_ok=True)

    score_file = os.path.join(args.output_dir, "scores.yaml")

    is_labeled = False
    # Load existing per-split scores so incremental runs skip already-done splits.
    all_scores = {}
    if os.path.exists(score_file):
        if args.ovr:
            print(f"Overwriting existing scores file at {score_file}.")
        else:
            with open(score_file) as f:
                all_scores = yaml.safe_load(f) or {}

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(
        "Device:",
        device,
        os.environ.get("CUDA_VISIBLE_DEVICES", "All") if device.type == "cuda" else "CPU",
    )

    print(f"{'Argument':<30} {'Value'}")
    print("-" * 50)
    for arg in vars(args):
        print(f"{arg:<30} {getattr(args, arg)}")
    print("-" * 50)

    # Determine which splits still need running.
    pending = []
    for data_jsonl in args.data_jsonl:
        stem = os.path.splitext(os.path.basename(data_jsonl))[0]
        split_dir = os.path.join(args.output_dir, stem)
        already_done = os.path.exists(os.path.join(split_dir, "final_result.jsonl"))
        if already_done and not args.ovr:
            print(f"Skipping {stem} (output already exists). Pass --ovr to rerun.")
        else:
            pending.append(data_jsonl)

    if not pending:
        print("All splits already scored. Pass --ovr to rerun.")
        return

    # Load the ORCA model once — reused for every split.
    if os.path.isfile(os.path.join(args.model_path, "lm", "adapter_config.json")):
        # LoRA case: load base model + adapters
        base_model_name = json.load(
            open(os.path.join(args.model_path, "lm", "adapter_config.json"))
        )["base_model_name_or_path"]
        model_dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
        lm = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=model_dtype,
            device_map=device.type,
            low_cpu_mem_usage=True,
        )
        lm = PeftModel.from_pretrained(
            lm,
            os.path.join(args.model_path, "lm"),
            low_cpu_mem_usage=True,
        )
        lm = lm.merge_and_unload()  # type: ignore
    else:
        # Full fine-tuning case: load complete fine-tuned model
        model_dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
        lm = AutoModelForCausalLM.from_pretrained(
            os.path.join(args.model_path, "lm"),
            device_map=device.type,
            low_cpu_mem_usage=True,
            dtype=model_dtype,
        )
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    else:
        # Checkpoint layout: best/model/lm, best/tokenizer
        # model_path points to best/model, so tokenizer is one level up
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(os.path.dirname(args.model_path), "tokenizer"),
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    scoring_model = model.ORCA.load_from_directory(args.model_path, lm=lm, device=device)
    collate_fn = data.CollateFn(tokenizer)
    # multinomial_llh expects raw integer ratings [1,5]; all other score types
    # use [0,1]-normalised floats.
    normalize_ratings = scoring_model.score_type != "multinomial"

    for data_jsonl in pending:
        stem = os.path.splitext(os.path.basename(data_jsonl))[0]
        split_dir = os.path.join(args.output_dir, stem)
        os.makedirs(split_dir, exist_ok=True)
        if not args.do_not_evaluate:
            is_labeled = _is_labeled(data_jsonl)

        inference_data = data.UnifiedAnnotationDataset(
            data_jsonl,
            skip_question=args.skip_question,
            skip_rationale=args.skip_rationale,
            add_transcript=args.add_transcript,
            normalize_ratings=normalize_ratings,
        )
        inference_loader = DataLoader(
            inference_data,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=0,
        )

        scoring_model.eval()
        all_metadata = []
        all_params = []
        all_prob1 = []
        all_variance = []
        all_ratings = []
        all_ground_truth_variances = []
        all_n_ratings = []
        with torch.inference_mode():
            for batch in tqdm.tqdm(inference_loader, desc=f"Inference [{stem}]"):
                batch = batch.to(device)
                if device.type == "cpu":
                    outputs = scoring_model(batch)
                else:
                    with torch.autocast(device.type, dtype=torch.bfloat16):
                        outputs = scoring_model(batch)
                all_metadata.extend(batch["metadata"])
                all_params.extend(outputs["params"].cpu().tolist())
                all_prob1.extend(outputs["prob1"].cpu().tolist())
                all_variance.extend(outputs["variance"].cpu().tolist())
                all_n_ratings.extend(batch["n_ratings"].cpu().tolist())
                if is_labeled:
                    all_ratings.extend(batch["average_labels"].cpu().tolist())
                    all_ground_truth_variances.extend(batch["label_variance"].cpu().tolist())

        # One row per input — original fields plus ORCA predictions
        with open(os.path.join(split_dir, "final_result.jsonl"), "w") as f:
            for i, (meta, param, prob1, variance) in enumerate(
                zip(all_metadata, all_params, all_prob1, all_variance)
            ):
                entry = {
                    **meta,
                    "rating_orca": prob1,
                    "variance_orca": variance,
                    "params": param,
                    "n_ratings": all_n_ratings[i],
                }
                if is_labeled:
                    entry["rating_ground_truth"] = all_ratings[i]
                    entry["variance_ground_truth"] = all_ground_truth_variances[i]
                f.write(json.dumps(entry) + "\n")

        if is_labeled:
            # For multinomial, ground-truth average_labels and label_variance are in
            # [1,5] / [0,16] space (normalize_ratings=False); scale them down to [0,1]
            # so MAE is comparable with beta/bernoulli. Predictions (prob1, variance)
            # are already in [0,1] space.
            gt_ratings = all_ratings
            gt_variances = all_ground_truth_variances
            if scoring_model.score_type == "multinomial":
                gt_ratings = [(r - 1) / 4 for r in all_ratings]
                gt_variances = [v / 16 for v in all_ground_truth_variances]

            # Variance metrics use only samples with >= 3 annotations (unbiased
            # estimate requires n >= 2; >= 3 provides a more reliable variance estimate).
            var_mask = [n >= 3 for n in all_n_ratings]
            n_var = sum(var_mask)
            gt_var_sub = [v for v, ok in zip(gt_variances, var_mask) if ok]
            pred_var_sub = [v for v, ok in zip(all_variance, var_mask) if ok]

            m = utils.compute_metrics(gt_ratings, all_prob1, gt_var_sub, pred_var_sub)

            metrics = {
                "n": len(all_ratings),
                "mean_mae": round(m["mean_mae"], 6),
                "rating_kendall_tau": round(m["mean_tau"], 6),
                "rating_spearman_correlation": round(m["mean_rho"], 6),
            }

            print("\n" + "=" * 40)
            print(f"{'Metric':<25} {'Value':>10}")
            print("-" * 40)
            print(f"{'MAE':<25} {m['mean_mae']:>10.6f}")
            print(f"{'Kendall τ (tau)':<25} {m['mean_tau']:>10.6f}")
            print(f"{'Spearman ρ (rho)':<25} {m['mean_rho']:>10.6f}")

            if scoring_model.score_type in ("beta", "multinomial"):
                print(f"{'Variance MAE':<25} {m['variance_mae']:>10.6f}  (n_var={n_var})")
                print(f"{'Variance Kendall τ':<25} {m['variance_tau']:>10.6f}")
                print(f"{'Variance Spearman ρ':<25} {m['variance_rho']:>10.6f}")
                metrics["n_var"] = n_var
                metrics["variance_mae"] = round(m["variance_mae"], 6)
                metrics["variance_kendall_tau"] = round(m["variance_tau"], 6)
                metrics["variance_spearman_correlation"] = round(m["variance_rho"], 6)

            print("=" * 40 + "\n")

            all_scores[stem] = metrics

    if all_scores:
        with open(score_file, "w") as f:
            yaml.dump(all_scores, f)
        print(f"Scores written to {score_file}")


def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="ORCA Inference Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained ORCA model directory (best/model)",
    )
    parser.add_argument(
        "--data_jsonl",
        type=str,
        nargs="+",
        required=True,
        help="One or more JSONL files to score. Results for each are saved in output_dir/<stem>/.",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for DataLoader"
    )

    parser.add_argument("--tokenizer_path", type=str, help="Path to the tokenizer directory.")
    parser.add_argument(
        "--skip_rationale",
        action="store_true",
        help="If set, the model will not use rationales for scoring.",
    )
    parser.add_argument(
        "--skip_question",
        action="store_true",
        help="If set, the model will not use questions for scoring.",
    )
    parser.add_argument(
        "--add_transcript",
        action="store_true",
        help="If set, the model will use transcript as additional context.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save inference results. Defaults to sub-dir: `inference_results` in model directory.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run inference on CPU (uses float32 instead of bfloat16).",
    )
    parser.add_argument(
        "--ovr",
        action="store_true",
        help="Overwrite existing scores file if it exists. Use with caution to avoid losing previous results.",
    )
    parser.add_argument(
        "--do_not_evaluate",
        action="store_true",
        help="do not compute evaluation metrics irrespective of whether the test set is labelled or not",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
