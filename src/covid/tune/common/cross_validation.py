from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

type TuningCV = StratifiedKFold | RepeatedStratifiedKFold


def get_cv_fold_counts(cv: TuningCV) -> tuple[int, int]:
    """Return the number of splits per repeat and the number of repeats."""
    n_repeats: int = getattr(cv, "n_repeats", 1)
    n_splits_per_repeat = cv.get_n_splits() // n_repeats
    return n_splits_per_repeat, n_repeats
