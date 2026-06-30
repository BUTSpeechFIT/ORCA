"""Author: Šimon Sedláček"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml
from scipy.stats import kendalltau, spearmanr


def clamp_orca_score(orca_score, orca_var, var_threshold):
    """Clamp ORCA score to 0 or 1 if variance is below threshold and score is extreme."""

    if var_threshold is None or orca_score is None or orca_var is None:
        raise ValueError("var_threshold, orca_score, and orca_var must not be None")

    if orca_var <= var_threshold:
        if orca_score < 0.125:
            return 0.0
        elif orca_score > 0.875:
            return 1.0

    return orca_score


def evaluate_orca(human_scores, orca_scores, human_vars=None, orca_vars=None):
    rho = spearmanr(human_scores, orca_scores).statistic
    tau = kendalltau(human_scores, orca_scores).statistic
    mae = np.mean(np.abs(np.array(human_scores) - np.array(orca_scores)))
    result = {"rho": float(rho), "tau": float(tau), "mae": float(mae)}
    if human_vars is not None and orca_vars is not None:
        result["variance_mae"] = float(np.mean(np.abs(np.array(human_vars) - np.array(orca_vars))))
    return result


def compute_composite_score(results, weights=(1.0, 1.0, 0.0)):
    """
    Combine rho, tau, and MAE into a single score (higher is better).
    Note: MAE has negative weight since lower MAE is better.

    Args:
        results: dict with 'rho', 'tau', 'mae'
        weights: tuple of (rho_weight, tau_weight, mae_weight)
    """
    return weights[0] * results["rho"] + weights[1] * results["tau"] + weights[2] * results["mae"]


def load_split(split_dir: Path) -> Tuple[List | None, List, List, List | None, List]:
    """Load human scores, orca scores, predicted variances, ground-truth variances,
    and annotation counts.

    Reads from final_result.jsonl — one row per input sample, in original order.
    Returns: (human, orca, orca_var, human_var, n_ratings)
      n_ratings: number of human annotations per sample; defaults to 99 if missing.
    """

    orca, orca_var, human, human_var, n_ratings = [], [], [], [], []
    with open(split_dir / "final_result.jsonl") as f:
        for line in f:
            row = json.loads(line)
            orca.append(row["rating_orca"])
            orca_var.append(row["variance_orca"])
            if "rating_ground_truth" in row:
                human.append(row["rating_ground_truth"])
            if "variance_ground_truth" in row:
                human_var.append(row["variance_ground_truth"])
            # default 99 for old files that pre-date n_ratings field
            n_ratings.append(row.get("n_ratings", 99))
    return human or None, orca, orca_var, human_var or None, n_ratings


def apply_clamp(
    human_scores, orca_scores, orca_vars, var_threshold, human_vars=None, orca_vars_eval=None
) -> dict:
    """Apply clamping with a fixed threshold and return evaluation metrics.

    orca_vars     — full list used to decide whether to clamp each score
    orca_vars_eval — optional filtered subset matched to human_vars for variance MAE;
                    defaults to orca_vars when not supplied.
    """
    clamped = [clamp_orca_score(s, v, var_threshold) for s, v in zip(orca_scores, orca_vars)]
    ev_ov = orca_vars_eval if orca_vars_eval is not None else orca_vars
    return evaluate_orca(human_scores, clamped, human_vars, ev_ov)


def find_threshold(
    human_scores, orca_scores, orca_vars, weights=(1.0, 1.0, 0.0)
) -> tuple[float, list]:
    """Estimate optimal var_threshold via hill-climb on composite score.

    Returns:
        best_threshold: float
        search_trace: list of (threshold, metrics_dict, composite_score)
    """
    var_threshold = 0.05  # starting point, search goes down first
    best_score = float("-inf")
    best_threshold = var_threshold
    no_improvement_count = 0
    use_factor_five = True
    going_down = True
    search_trace = []

    while True:
        results = apply_clamp(human_scores, orca_scores, orca_vars, var_threshold)
        composite_score = compute_composite_score(results, weights)
        search_trace.append((var_threshold, results, composite_score))

        if composite_score > best_score:
            best_score = composite_score
            best_threshold = var_threshold
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Going down: stop after 2 non-improving steps, switch to going up
        if going_down and no_improvement_count >= 2:
            going_down = False
            var_threshold = best_threshold
            use_factor_five = True
            no_improvement_count = 0
            continue

        # Going up: stop after 6 non-improving steps
        if not going_down and no_improvement_count >= 6:
            break

        if going_down:
            factor = 5 if use_factor_five else 2
            var_threshold = var_threshold / factor
            use_factor_five = not use_factor_five
            if var_threshold < 1e-6:
                going_down = False
                var_threshold = best_threshold
                use_factor_five = True
                no_improvement_count = 0
        else:
            var_threshold = var_threshold * 1.15
            use_factor_five = not use_factor_five
            if var_threshold > 0.05:
                break

    return best_threshold, search_trace


def _print_comparison(split: str, baseline: dict, clamped: dict) -> None:
    has_var = "variance_mae" in baseline
    hdr = f"    {'':10s}  {'ρ':>8}  {'τ':>8}  {'MAE':>8}"
    if has_var:
        hdr += f"  {'varMAE':>8}"
    print(f"\n  {split}:")
    print(hdr)
    for label, m in [("baseline", baseline), ("clamped", clamped)]:
        row = f"    {label:10s}  {m['rho']:8.4f}  {m['tau']:8.4f}  {m['mae']:8.6f}"
        if has_var:
            row += f"  {m['variance_mae']:8.6f}"
        print(row)
    delta_rho = clamped["rho"] - baseline["rho"]
    delta_tau = clamped["tau"] - baseline["tau"]
    delta_mae = clamped["mae"] - baseline["mae"]
    row = f"    {'delta':10s}  {delta_rho:+8.4f}  {delta_tau:+8.4f}  {delta_mae:+8.6f}"
    if has_var:
        row += f"  {clamped['variance_mae'] - baseline['variance_mae']:+8.6f}"
    print(row)


def _read_score_type(start: Path) -> Optional[str]:
    """Walk upward from start to find args.yaml and read score_type."""
    candidate = start.resolve()
    for _ in range(6):
        candidate = candidate.parent
        f = candidate / "args.yaml"
        if f.exists():
            try:
                data = yaml.safe_load(f.read_text()) or {}
                return data.get("score_type")
            except Exception:
                return None
    return None


def _scale_gt(
    human: "List | None",
    human_var: "List | None",
    score_type: Optional[str],
) -> "tuple[List | None, List | None]":
    """Scale multinomial ground truth to [0,1] to match prediction space.

    rating_ground_truth  ∈ [1,5]  → (r-1)/4  ∈ [0,1]
    variance_ground_truth ∈ [0,16] → v/16     ∈ [0,0.25]

    beta and bernoulli annotations are already in [0,1]; this is a no-op for them.
    """
    if score_type != "multinomial":
        return human, human_var
    scaled_human = [(r - 1) / 4 for r in human] if human is not None else None
    scaled_var = [v / 16 for v in human_var] if human_var is not None else None
    return scaled_human, scaled_var


def main():
    args = parse_arguments()

    weights = tuple(args.weights)

    # ── apply_dir mode: apply a pre-determined threshold to a single split ────
    if args.apply_dir is not None:
        if args.var_threshold is None:
            raise SystemExit("--var_threshold is required when --apply_dir is used")
        apply_dir = Path(args.apply_dir)
        score_type = _read_score_type(apply_dir)
        threshold = args.var_threshold
        human, orca, orca_var, human_var, n_ratings = load_split(apply_dir)
        human, human_var = _scale_gt(human, human_var, score_type)
        print(f"Applying var_threshold={threshold:.8f} to {apply_dir}")
        if score_type == "multinomial":
            print("  score_type=multinomial: ground-truth scaled to [0,1]")
        clamped_scores = [clamp_orca_score(s, v, threshold) for s, v in zip(orca, orca_var)]
        out: dict = {"n": len(orca), "var_threshold": round(float(threshold), 8)}
        if human is not None:
            var_mask = [n >= 3 for n in n_ratings]
            n_var = sum(var_mask)
            hv_sub = [v for v, ok in zip(human_var, var_mask) if ok] if human_var else None
            ov_sub = [v for v, ok in zip(orca_var, var_mask) if ok]
            baseline = evaluate_orca(human, orca, hv_sub, ov_sub)
            metrics = evaluate_orca(human, clamped_scores, hv_sub, ov_sub)
            print("\nResults:")
            _print_comparison(apply_dir.name, baseline, metrics)
            out["rating_kendall_tau"] = round(metrics["tau"], 6)
            out["rating_spearman_correlation"] = round(metrics["rho"], 6)
            out["mean_mae"] = round(metrics["mae"], 6)
            if "variance_mae" in metrics:
                out["n_var"] = n_var
                out["variance_mae"] = round(metrics["variance_mae"], 6)
        out_path = apply_dir / "score_clamp.yaml"
        with open(out_path, "w") as f:
            yaml.dump(out, f)
        print(f"\nWrote {out_path}")
        return

    if args.input_dir is None:
        raise SystemExit("One of --input_dir or --apply_dir is required")

    input_dir = Path(args.input_dir)
    score_type = _read_score_type(input_dir)
    if score_type == "multinomial":
        print("score_type=multinomial: ground-truth will be scaled to [0,1]")

    dev_dir = input_dir / args.dev_folder
    test_dirs = [input_dir / name for name in args.test_folders]

    if not dev_dir.exists():
        raise FileNotFoundError(f"{args.dev_folder} not found in {input_dir}")
    for td in test_dirs:
        if not td.exists():
            raise FileNotFoundError(f"{td.name} not found in {input_dir}")

    dev_human, dev_orca, dev_var, dev_human_var, dev_n_ratings = load_split(dev_dir)
    dev_human, dev_human_var = _scale_gt(dev_human, dev_human_var, score_type)

    dev_var_mask = [n >= 3 for n in dev_n_ratings]
    dev_hv_sub = [v for v, ok in zip(dev_human_var, dev_var_mask) if ok] if dev_human_var else None
    dev_ov_sub = [v for v, ok in zip(dev_var, dev_var_mask) if ok]

    # ── Threshold estimation or direct use ──────────────────────────────────
    if args.var_threshold is not None:
        threshold = args.var_threshold
        print(f"Using provided var_threshold: {threshold}")
    else:
        print("Estimating var_threshold from dev split...")
        threshold, search_trace = find_threshold(dev_human, dev_orca, dev_var, weights=weights)
        print(f"Best var_threshold: {threshold:.6f}")
        if args.verbose:
            print("\nSearch trace:")
            for t, m, cs in search_trace:
                print(
                    f"  {t:.6f}  rho={m['rho']:.4f}  tau={m['tau']:.4f}  mae={m['mae']:.6f}  composite={cs:.6f}"
                )

        # Guard: if clamping hurts dev performance, disable it entirely.
        dev_baseline_for_check = evaluate_orca(dev_human, dev_orca, dev_hv_sub, dev_ov_sub)
        dev_clamped_for_check = apply_clamp(
            dev_human, dev_orca, dev_var, threshold, dev_hv_sub, dev_ov_sub
        )
        if compute_composite_score(dev_clamped_for_check, weights) <= compute_composite_score(
            dev_baseline_for_check, weights
        ):
            print(
                f"WARNING: clamping (threshold={threshold:.6f}) does not improve dev composite score "
                f"({compute_composite_score(dev_clamped_for_check, weights):.4f} <= "
                f"{compute_composite_score(dev_baseline_for_check, weights):.4f}). "
                "Setting threshold=0 (no clamping)."
            )
            threshold = 0.0

    # ── Evaluate ─────────────────────────────────────────────────────────────
    dev_baseline = evaluate_orca(dev_human, dev_orca, dev_hv_sub, dev_ov_sub)
    dev_clamped = apply_clamp(dev_human, dev_orca, dev_var, threshold, dev_hv_sub, dev_ov_sub)

    print("\nResults:")
    _print_comparison(dev_dir.name, dev_baseline, dev_clamped)

    test_results: list[tuple[Path, dict, int, int]] = []
    for td in test_dirs:
        t_human, t_orca, t_var, t_human_var, t_n_ratings = load_split(td)
        t_human, t_human_var = _scale_gt(t_human, t_human_var, score_type)
        t_var_mask = [n >= 3 for n in t_n_ratings]
        t_n_var = sum(t_var_mask)
        t_hv_sub = [v for v, ok in zip(t_human_var, t_var_mask) if ok] if t_human_var else None
        t_ov_sub = [v for v, ok in zip(t_var, t_var_mask) if ok]
        t_baseline = evaluate_orca(t_human, t_orca, t_hv_sub, t_ov_sub)
        t_clamped = apply_clamp(t_human, t_orca, t_var, threshold, t_hv_sub, t_ov_sub)
        _print_comparison(td.name, t_baseline, t_clamped)
        test_results.append((td, t_clamped, len(t_orca), t_n_var))

    print("\n")
    # ── Write score_clamp.yaml ────────────────────────────────────────────────
    dev_n_var = sum(dev_var_mask)
    for split_dir, metrics, n, n_var in [
        (dev_dir, dev_clamped, len(dev_orca), dev_n_var),
        *test_results,
    ]:
        out = {
            "n": n,
            "var_threshold": round(float(threshold), 8),
            "rating_kendall_tau": round(metrics["tau"], 6),
            "rating_spearman_correlation": round(metrics["rho"], 6),
            "mean_mae": round(metrics["mae"], 6),
        }
        if "variance_mae" in metrics:
            out["n_var"] = n_var
            out["variance_mae"] = round(metrics["variance_mae"], 6)
        out_path = split_dir / "score_clamp.yaml"
        with open(out_path, "w") as f:
            yaml.dump(out, f)
        print(f"Wrote {out_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Calibrate ORCA scores by variance-gated clamping. "
        "Threshold is estimated on dev and applied to test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Experiment results dir containing the dev and test split subdirectories.",
    )
    parser.add_argument(
        "--dev_folder",
        type=str,
        default="dev",
        help="Subdirectory name inside --input_dir used for threshold estimation.",
    )
    parser.add_argument(
        "--test_folders",
        type=str,
        nargs="+",
        default=["test", "test_mmau_pro_248", "test_mmau_pro_41"],
        help="Subdirectory name(s) inside --input_dir to apply the threshold to.",
    )
    parser.add_argument(
        "--apply_dir",
        type=str,
        default=None,
        help="Apply a pre-determined --var_threshold to this single result directory "
        "(must contain final_result.jsonl). Writes score_clamp.yaml there. "
        "Mutually exclusive with --input_dir; requires --var_threshold.",
    )
    parser.add_argument(
        "--var_threshold",
        type=float,
        default=None,
        help="If provided, skip estimation and apply this threshold directly.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        metavar=("W_RHO", "W_TAU", "W_MAE"),
        default=[1.0, 1.0, 0.0],
        help="Weights for composite score: w_rho * rho + w_tau * tau + w_mae * mae.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full threshold search trace.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    np.seterr(all="raise")
    main()
