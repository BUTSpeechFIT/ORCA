"""Command-line interface for ORCA-Score.

Authors: Bolaji Yusuf, Santosh Kesiraju
"""


def orca_train():
    """Entry point for orca-train command."""
    from orca_score.train import main

    main()


def orca_infer():
    """Entry point for orca-infer command."""
    from orca_score.infer import main

    main()
