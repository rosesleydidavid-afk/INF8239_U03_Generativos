"""Métricas de reconstrucción, diversidad y memoria. Solo NumPy y scikit-learn (sin TensorFlow)."""
from __future__ import annotations

import numpy as np


def flatten(images: np.ndarray) -> np.ndarray:
    return np.asarray(images, dtype="float32").reshape(len(images), -1)


def mse_per_image(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    if original.shape != reconstructed.shape:
        raise ValueError(f"Formas distintas: {original.shape} y {reconstructed.shape}")
    return ((flatten(original) - flatten(reconstructed)) ** 2).mean(axis=1)


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(mse_per_image(original, reconstructed).mean())


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """PSNR en dB para imágenes en [0, 1]; se acota el error para evitar infinitos."""
    return float(10 * np.log10(1.0 / max(mse(original, reconstructed), 1e-12)))


def per_class_mse(original, reconstructed, labels, n_classes: int = 10) -> list[float]:
    errors = mse_per_image(original, reconstructed)
    return [float(errors[labels == c].mean()) if (labels == c).any() else float("nan") for c in range(n_classes)]


def fixed_indices(n_total: int, size: int, seed: int) -> np.ndarray:
    """Selección reproducible y no manual de ejemplos para los paneles."""
    if size > n_total:
        raise ValueError("No hay suficientes ejemplos para el panel")
    return np.sort(np.random.default_rng(seed).choice(n_total, size=size, replace=False))


def pair_with_different_labels(labels: np.ndarray, seed: int) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(labels)))
    candidates = np.where(labels != labels[first])[0]
    if len(candidates) == 0:
        raise ValueError("Se necesitan al menos dos clases")
    return first, int(rng.choice(candidates))


def nearest_neighbor(queries: np.ndarray, reference: np.ndarray, chunk: int = 500):
    """Distancia euclídea e índice del vecino más cercano de cada consulta dentro de `reference`."""
    q, r = flatten(queries), flatten(reference)
    r_norm = (r ** 2).sum(axis=1)
    distances = np.empty(len(q), dtype="float64")
    indices = np.empty(len(q), dtype=int)
    for start in range(0, len(q), chunk):
        block = q[start:start + chunk]
        approx = (block ** 2).sum(axis=1)[:, None] - 2 * block @ r.T + r_norm[None, :]
        best = approx.argmin(axis=1)
        indices[start:start + chunk] = best
        distances[start:start + chunk] = np.linalg.norm(block - r[best], axis=1)
    return distances, indices


def mean_pairwise_distance(images: np.ndarray) -> float:
    """Distancia euclídea media entre pares distintos: mide cuán variado es un conjunto."""
    x = flatten(images).astype("float64")
    if len(x) < 2:
        return 0.0
    squared = (x ** 2).sum(axis=1)
    d2 = np.maximum(squared[:, None] + squared[None, :] - 2 * x @ x.T, 0)
    upper = np.triu_indices(len(x), k=1)
    return float(np.sqrt(d2[upper]).mean())


def class_entropy(counts: np.ndarray) -> float:
    """Entropía (nats) del reparto de clases. Máximo ln(10) = 2.303 cuando las 10 clases aparecen por igual."""
    counts = np.asarray(counts, dtype="float64")
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def fraction_below(values: np.ndarray, threshold: float) -> float:
    return float((np.asarray(values) < threshold).mean())


def pca_reconstruction(train: np.ndarray, test: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    """Línea base lineal: proyecta y reconstruye con PCA ajustado solo con entrenamiento."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=seed).fit(flatten(train))
    rebuilt = pca.inverse_transform(pca.transform(flatten(test))).clip(0, 1)
    return rebuilt.reshape(test.shape).astype("float32")
