import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests


def association_summary(loci_data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    # fmt:off
    return (
        loci_data
        .apply(lambda locus: locus_with_target_association(locus, target))
        .T
        .infer_objects()
        .assign(q_value=lambda df: multipletests(df["p_value"], method="fdr_bh")[1])
        .sort_values(["q_value", "cramers_v"], ascending=[True, False])
    )
    # fmt:on


def locus_with_target_association(locus: pd.Series, target: pd.Series) -> pd.Series:
    table = pd.crosstab(locus, target)
    chi2, p_value, _, expected = chi2_contingency(table)

    n = table.to_numpy().sum()
    min_dimension = min(table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dimension)) if min_dimension > 0 else np.nan

    return pd.Series(
        {
            "chi2": chi2,
            "p_value": p_value,
            "cramers_v": cramers_v,
            "min_expected_count": expected.min(),
            "sparse_table": (expected < 5).any(),
        },
        dtype=float,
    )


def loci_correlation(
    association_summary_data: pd.DataFrame, loci_data: pd.DataFrame, max_loci: int = 50
) -> pd.DataFrame:
    top_loci = association_summary_data.head(max_loci).index
    return loci_data[top_loci].corr(method="spearman")


def strongest_pairs(
    loci_correlations: pd.DataFrame, correlation_threshold: float = 0.8
) -> pd.DataFrame:
    upper_mask = np.triu(np.ones_like(loci_correlations, dtype=bool), k=1)
    return (
        loci_correlations.where(upper_mask)
        .stack()
        .rename("correlation")
        .reset_index()
        .rename(columns={"level_0": "locus_1", "level_1": "locus_2"})
        .assign(absolute_correlation=lambda data: data["correlation"].abs())
        .query(f"absolute_correlation >= {correlation_threshold}")
        .sort_values("absolute_correlation", ascending=False)
    )
