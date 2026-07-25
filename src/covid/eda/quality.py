import pandas as pd


def missing_summary_by_chromosome(loci_data: pd.DataFrame) -> pd.DataFrame:
    chromosomes = loci_data.columns.str.extract(r"^(chr(?:\d+|X|Y|M))", expand=False)

    locus_missing_rate = pd.DataFrame(
        {
            "chromosome": chromosomes,
            "missing_rate": loci_data.isna().mean().to_numpy() * 100,
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
