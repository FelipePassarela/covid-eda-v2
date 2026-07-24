import pandas as pd


def missing_summary_by_chromosome(
    df: pd.DataFrame, loci_columns: list[str]
) -> pd.DataFrame:
    locus_columns = pd.Series(loci_columns)
    chromosomes = locus_columns.str.extract(r"^(chr(?:\d+|X|Y|M))", expand=False)

    locus_missing_rate = pd.DataFrame(
        {
            "chromosome": chromosomes,
            "missing_rate": df[locus_columns].isna().mean().to_numpy() * 100,
        }
    )

    chr_summary = (
        locus_missing_rate.groupby("chromosome")
        .agg(
            loci_count=("missing_rate", "size"),
            missing_mean=("missing_rate", "mean"),
            missing_std=("missing_rate", "std"),
        )
        .sort_values("missing_mean", ascending=False)
        .round(2)
    )
    return chr_summary
