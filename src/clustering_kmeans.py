# src/clustering_kmeans.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass(frozen=True)
class KMeansConfig:
    k_min: int = 2
    k_max: int = 12
    random_state: int = 42
    init: str = "k-means++"
    n_init: int = 10
    max_iter: int = 300


def sanity_check_scaled(
    df: pd.DataFrame, 
    tol_mean: float = 0.15, 
    tol_std: float = 0.15
) -> dict:
    """
    Verifica (aproximadamente) se o dataframe está padronizado (Z-Score).
    Retorna um dicionário com métricas (não interrompe execução).
    """
    means = df.mean(numeric_only=True)
    stds = df.std(numeric_only=True)

    return {
        "mean_abs_mean": float(means.abs().mean()),
        "mean_abs_std_minus_1": float((stds - 1).abs().mean()),
        "ok_mean": bool(means.abs().mean() <= tol_mean),
        "ok_std": bool((stds - 1).abs().mean() <= tol_std),
    }


def compute_elbow_inertia(
    X: pd.DataFrame,
    k_values: Iterable[int],
    random_state: int = 42,
    init: str = "k-means++",
    n_init: int = 10,
    max_iter: int = 300,
) -> pd.DataFrame:
    """
    Elbow/WCSS: retorna dataframe com (k, inertia).
    """
    rows = []
    for k in k_values:
        model = KMeans(
            n_clusters=int(k),
            random_state=random_state,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
        )
        model.fit(X)
        rows.append({"k": int(k), "inertia": float(model.inertia_)})
    return pd.DataFrame(rows)


def compute_silhouette(
    X: pd.DataFrame,
    k_values: Iterable[int],
    random_state: int = 42,
    init: str = "k-means++",
    n_init: int = 10,
    max_iter: int = 300,
) -> pd.DataFrame:
    """
    Silhouette médio: retorna dataframe com (k, silhouette).
    """
    rows = []
    for k in k_values:
        k = int(k)
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
        )
        labels = model.fit_predict(X)

        score = silhouette_score(X, labels)
        rows.append({"k": k, "silhouette": float(score)})

    return pd.DataFrame(rows)


def fit_kmeans(
    X: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
    init: str = "k-means++",
    n_init: int = 10,
    max_iter: int = 300,
) -> Tuple[KMeans, np.ndarray]:
    """
    Treina KMeans final e retorna (modelo, labels).
    """
    model = KMeans(
        n_clusters=int(n_clusters),
        random_state=random_state,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
    )
    labels = model.fit_predict(X)
    return model, labels


def make_cluster_profile(
    df_scaled: pd.DataFrame,
    labels: np.ndarray,
    cluster_col: str = "cluster",
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Gera perfil por cluster (média ou mediana) e tamanho do cluster.
    """
    df_out = df_scaled.copy()
    df_out[cluster_col] = labels

    if agg not in {"mean", "median"}:
        raise ValueError("agg deve ser 'mean' ou 'median'.")

    if agg == "mean":
        prof = df_out.groupby(cluster_col).mean(numeric_only=True)
    else:
        prof = df_out.groupby(cluster_col).median(numeric_only=True)

    sizes = df_out[cluster_col].value_counts().sort_index()
    prof.insert(0, "cluster_size", sizes.values)
    return prof.reset_index()


def plot_elbow(elbow_df: pd.DataFrame, title: str = "Elbow (Inertia)") -> None:
    plt.figure()
    plt.plot(elbow_df["k"], elbow_df["inertia"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia (WCSS)")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_silhouette(sil_df: pd.DataFrame, title: str = "Silhouette Score") -> None:
    plt.figure()
    plt.plot(sil_df["k"], sil_df["silhouette"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette (médio)")
    plt.title(title)
    plt.grid(True)
    plt.show()


def save_current_figure(path: str | Path, dpi: int = 150) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight") 