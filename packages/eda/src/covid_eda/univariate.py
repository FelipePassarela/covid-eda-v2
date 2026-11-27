import numpy as np
import pandas as pd


def imbalance_summary(
    df: pd.DataFrame,
    categorical_columns: list[str],
    rare_threshold: float = 0.01,
) -> pd.DataFrame:
    rows = []

    for column in categorical_columns:
        counts = df[column].value_counts()
        proportions = counts / counts.sum()

        entropy = -(proportions * np.log2(proportions)).sum()
        maf = alleles_summary(df[column]).get("maf", np.nan)
        dominant_rate = proportions.iloc[0] if not proportions.empty else np.nan
        minority_rate = proportions.iloc[-1] if not proportions.empty else np.nan
        chromosomes = column.split("_")[0] if "_" in column else None

        rows.append(
            {
                "feature": column,
                "dominant_rate": dominant_rate,
                "minority_rate": minority_rate,
                "minority_count": counts.min(),
                "maf": maf,
                "rare_categories": proportions.lt(rare_threshold).sum(),
                "entropy": entropy,
                "chromosome": chromosomes,
            }
        )

    summary = (
        pd.DataFrame(rows)
        .set_index("feature")
        .sort_values(by="entropy", ascending=True)
    )
    return summary


def alleles_summary(genotypes: pd.Series) -> pd.Series:
    genotypes = pd.to_numeric(genotypes, errors="coerce").dropna()

    if genotypes.empty:
        return pd.Series({"maf": np.nan})

    total_alleles = 2 * len(genotypes)
    alternative_allele_count = genotypes.sum()
    alternative_af = alternative_allele_count / total_alleles

    maf = min(alternative_af, 1 - alternative_af)

    return pd.Series({"maf": maf})


def chromosomes_imbalance_summary(loci_data: pd.DataFrame) -> pd.DataFrame:
    loci_columns = loci_data.columns.to_list()
    summary = imbalance_summary(loci_data, loci_columns)
    return (
        summary.groupby("chromosome")
        .agg(
            n_features=("dominant_rate", "size"),
            mean_dominant_rate=("dominant_rate", "mean"),
            std_dominant_rate=("dominant_rate", "std"),
            mean_entropy=("entropy", "mean"),
            mean_maf=("maf", "mean"),
            near_constant_rate=("dominant_rate", lambda values: values.ge(0.97).mean()),
        )
        .sort_values(by="mean_maf", ascending=True)
    )
