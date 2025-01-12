"""Provides support for Qiime formats."""

from os import path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
from statsmodels.stats.multitest import fdrcorrection
from zipfile import ZipFile
from ruamel.yaml import YAML
from plotnine import *
from tempfile import TemporaryDirectory


_ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species", "strain"]
yaml = YAML()


def metadata(artifact):
    """Read metadata from a Qiime 2 artifact."""
    with ZipFile(artifact) as zf:
        files = zf.namelist()
        meta = [fi for fi in files if "metadata.yaml" in fi and "provenance" not in fi]
        if len(meta) == 0:
            raise ValueError("%s is not a valid Qiime 2 artifact :(" % artifact)
        with zf.open(meta[0]) as mf:
            meta = yaml.load(mf)
        return meta


def load_qiime_feature_table(artifact):
    """Load a feature table from a Qiime 2 artifact."""
    try:
        import biom
    except ImportError:
        raise ImportError(
            "Reading Qiime 2 FeatureTables requires the `biom-format` package."
            "You can install it with:\n pip install numpy Cython\n"
            "pip install biom-format"
        )
    meta = metadata(artifact)
    if not meta["type"].startswith("FeatureTable["):
        raise ValueError("%s is not a Qiime 2 FeatureTable :(" % artifact)
    uuid = meta["uuid"]
    with ZipFile(artifact) as zf, TemporaryDirectory(prefix="micom_") as td:
        zf.extract(uuid + "/data/feature-table.biom", str(td))
        table = biom.load_table(path.join(str(td), uuid, "data", "feature-table.biom"))
    return table


def load_qiime_taxonomy(artifact):
    """Load taxonomy feature data from a Qiime 2 artifact."""
    meta = metadata(artifact)
    if not meta["type"].startswith("FeatureData[Taxonomy]"):
        raise ValueError("%s is not a Qiime 2 FeatureData object :(" % artifact)
    uuid = meta["uuid"]
    with ZipFile(artifact) as zf, TemporaryDirectory(prefix="micom_") as td:
        zf.extract(uuid + "/data/taxonomy.tsv", str(td))
        taxa = pd.read_csv(
            path.join(str(td), uuid, "data", "taxonomy.tsv"), sep="\t", index_col=0
        )["Taxon"]
    return taxa


def build_from_qiime(
    abundance,
    taxonomy: pd.Series,
    collapse_on=["kingdom", "phylum", "class", "order", "family", "genus","species"],
) -> pd.DataFrame:
    """Format qiime results into a long DataFrame."""
    taxa = taxonomy.str.replace("[\\w_]+__|\\[|\\]", "", regex=True)
    taxa = taxa.str.split(";\\s*", expand=True).replace("", None)
    taxa.columns = _ranks[0 : taxa.shape[1]]
    taxa["taxid"] = taxonomy.index
    taxa.index == taxa.taxid

    if isinstance(collapse_on, str):
        collapse_on = [collapse_on]

    ranks = [r for r in collapse_on if r in taxa.columns]
    taxa["mapping_ranks"] = taxa[ranks].apply(
        lambda s: "|".join(s.astype("str")), axis=1
    )

    abundance = (
        abundance.collapse(
            lambda id_, x: taxa.loc[id_, "mapping_ranks"],
            axis="observation",
            norm=False,
        )
        .to_dataframe(dense=True)
        .T
    )
    abundance["sample_id"] = abundance.index

    abundance = abundance.melt(
        id_vars="sample_id", var_name="mapping_ranks", value_name="reads"
    )
    abundance = pd.merge(
        abundance[abundance.reads > 0.0],
        taxa[ranks + ["mapping_ranks"]].drop_duplicates(),
        on="mapping_ranks",
    )
    abundance.rename(columns={"mapping_ranks": "full_taxonomy"}, inplace=True)
    depth = abundance.groupby("sample_id").reads.sum()
    abundance["relative"] = abundance.reads / depth[abundance.sample_id].values

    return abundance


def qiime_to_dataframe(
    feature_table,
    taxonomy,
    collapse_on=["kingdom", "phylum", "class", "order", "family", "genus","species"],
):
    """Build a DataFrame from Qiime 2 data.
    Parameters
    ----------
    feature_table : str
        Path to a Qiime 2 FeatureTable artifact.
    taxonomy : str
        Path to a Qiime 2 FeatureData[Taxonomy] artifact.
    collapse_on : str or List[str]
        The taxa ranks to collapse on. This will dictate how strict the database
        matching will be as well.
    Returns
    -------
    pd.DataFrame
        A micom taxonomy containing abundances and taxonomy calls in long
        format.
    """
    table = load_qiime_feature_table(feature_table)
    taxonomy = load_qiime_taxonomy(taxonomy)

    return build_from_qiime(table, taxonomy, collapse_on)


def filter_taxa(df, min_reads=2, min_prevalence=0.5):
    """Filter an abundance DataFrame by reads and prevalence."""
    n = df.sample_id.nunique()
    prevalence = df[df.reads > 0].full_taxonomy.value_counts() / n
    good_p = prevalence[prevalence >= min_prevalence].index
    mean_reads = df.groupby("full_taxonomy").reads.mean()
    good_r = mean_reads[mean_reads >= min_reads].index
    df = df.copy()
    return df[(df.full_taxonomy.isin(good_r)) & (df.full_taxonomy.isin(good_p))]


def clr(df, pseudocount=0.5):
    """Calculate the centered log transform for a table."""
    df = df.copy()
    log_gm = df.groupby("sample_id").apply(
        lambda df: np.log2(df.reads + pseudocount).mean()
    )
    df["clr"] = np.log2(df.reads + 0.5) - log_gm[df.sample_id].values
    return df


def mwstats(x, y):
    try:
        test = mannwhitneyu(x, y)
    except Exception:
        test = (float("nan"), float("nan"))

    se = np.sqrt(
        (np.std(x) / np.sqrt(len(x))) ** 2 + (np.std(y) / np.sqrt(len(y))) ** 2
    )
    return pd.Series(
        [test[0], y.mean() - x.mean(), se, len(x) + len(y), test[1]],
        index=["U_statistic", "log2_fold_change", "standard_error", "n", "p"],
    )

def mwtests(
    table,
    taxa,
    metadata,
    metadata_column,
    control_group="healthy",
    collapse_on=["kingdom", "phylum", "class", "order", "family", "genus"],
    min_reads=2,
    min_prevalence=0.5,
):
    """Calculate the Mann-Whitney p values and log2 fold-change between cases and controls.
    Parameters
    ----------
    table : str
        Path to a Qiime2 feature table.
    taxa : str
        Path to a Qiime2 taxonomy.
    metadata : str
        Path to Qiime2 metadata.
    metadata_column : str
        Name of the column denoting cases and controls.
    control_group : str
        Name of the control group.
    collapse_on : list
        Taxonomy ranks to collapse on.
    min_reads : float
        Minimum mean reads for a taxon to be considered.
    min_prevalence : float
        Minimum prevalence (% of samples) for a taxon to be considered.
    Returns
    -------
    pandas.DataFrame
        A table containing the taxon, the log2 fold change between case and control, the uncorrected and
        corrected p values.
    """
    ab = qiime_to_dataframe(table, taxa, collapse_on=collapse_on)
    ab = ab.dropna(subset=collapse_on)
    ab = clr(filter_taxa(ab, min_reads=min_reads, min_prevalence=min_prevalence))
    meta = pd.read_csv(metadata, sep="\t")
    meta.rename(columns={meta.columns[0]: "sample_id"}, inplace=True)
    ab = pd.merge(ab, meta, on="sample_id")
    results = []
    results = (
        ab.groupby(collapse_on)
        .apply(
            lambda df: mwstats(
                df.clr[df[metadata_column] == control_group],
                df.clr[df[metadata_column] != control_group],
            )
        )
        .reset_index()
    )
    return results


def clr_transform_individual(row):
    row = row + 0.5
    geometric_mean = np.exp(np.mean(np.log(row)))
    clr_values = np.log(row / geometric_mean)
    return clr_values

def effectsize(group1, group2):
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1), np.std(group2)
    
    pooled_std = np.sqrt(((len(group1) - 1) * std1**2 + (len(group2) - 1) * std2**2) / (len(group1) + len(group2) - 2))
    cohens_d = (mean1 - mean2) / pooled_std
    return cohens_d