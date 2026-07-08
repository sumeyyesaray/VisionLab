import pandas as pd


def subsample_per_class(
    dataframe: pd.DataFrame,
    n_per_class: int,
    label_col: str = "label",
    seed: int = 42,
) -> pd.DataFrame:
    """Take up to `n_per_class` random rows per label — for quick, stratified
    experiments on a small slice of a dataset without touching the full
    pipeline (see notebooks/06_first_real_training_run.ipynb)."""
    parts = [
        group.sample(n=min(n_per_class, len(group)), random_state=seed)
        for _, group in dataframe.groupby(label_col)
    ]
    return pd.concat(parts, ignore_index=True)
