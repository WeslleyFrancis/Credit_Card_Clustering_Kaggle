from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (18, 14),
    bins: int = 30,
    tight_layout: bool = True,
) -> None:
    ax = df.hist(figsize=figsize, bins=bins)
    if tight_layout:
        plt.tight_layout()
    plt.show()

def plot_boxplot_all_features(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (18, 10),
    rotate_xticks: int = 90,
) -> None:
    plt.figure(figsize=figsize)
    sns.boxplot(data=df)
    plt.xticks(rotation=rotate_xticks)
    plt.show()

def plot_corr_heatmap(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (14, 10),
    cmap: str = "coolwarm",
    center: float = 0.0,
    annot: bool = False,
) -> None:
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap=cmap, center=center, annot=annot)
    plt.show()