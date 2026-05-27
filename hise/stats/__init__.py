"""Statistical inference utilities for HISE headline tables."""
from hise.stats.bootstrap import (
    bootstrap_mean_ci,
    cohens_d,
    effect_size_tag,
    holm_bonferroni,
    paired_bootstrap_ci,
    paired_permutation_pvalue,
)

__all__ = [
    "bootstrap_mean_ci",
    "cohens_d",
    "effect_size_tag",
    "holm_bonferroni",
    "paired_bootstrap_ci",
    "paired_permutation_pvalue",
]
