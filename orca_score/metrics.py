"""Agreement between ORCA scores and human ratings."""

from orca_score.clamp import evaluate_orca
from orca_score.pairs import clean_ratings


def mean_human_rating(ratings: list) -> float:
    """Average of the human ratings, scaled from the 1-5 scale to [0, 1].

    Args:
        ratings: Valid ratings (numbers from 1 to 5).

    Returns:
        The scaled mean, (mean - 1) / 4.
    """
    return (sum(ratings) / len(ratings) - 1) / 4


def agreement_metrics(
    rows: list, score_key: str = "orca_score", clamped_key: str = "orca_score_clamped"
) -> dict:
    """Spearman, Kendall and MAE between ORCA scores and the mean human ratings.

    Args:
        rows: Scored rows that all have human ratings.
        score_key: Key of the raw ORCA score.
        clamped_key: Key of the clamped score; metrics for it are added if rows have it.

    Returns:
        Dict with n, "raw" and, if present, "clamped"; each has rho, tau and mae.
    """
    human = [mean_human_rating(clean_ratings(r["ratings"])) for r in rows]
    metrics = {"n": len(rows), "raw": evaluate_orca(human, [r[score_key] for r in rows])}
    if rows and clamped_key in rows[0]:
        metrics["clamped"] = evaluate_orca(human, [r[clamped_key] for r in rows])
    return metrics
