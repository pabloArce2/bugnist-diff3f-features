"""Descriptor loading, PCA colouring and shared k-means clustering.

Descriptors are L2-normalised before PCA and k-means unless normalize=False.
"Shared" means that one PCA basis (and one k-means model) is fitted on samples
from every compared shape, so a colour or cluster id means the same thing on
all of them.
"""

import colorsys

import numpy as np
import torch
from sklearn.cluster import KMeans

CLUSTER_SPACES = ("features", "pca")


def load_features(path):
    """A [N, D] float32 descriptor tensor from a .pt file (a tensor or a dict holding one)."""
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, torch.Tensor):
        features = loaded
    elif isinstance(loaded, dict):
        for key in ("features", "feat", "x"):
            if isinstance(loaded.get(key), torch.Tensor):
                features = loaded[key]
                break
        else:
            raise ValueError(f"{path} is a dict but has no tensor under features/feat/x.")
    else:
        raise ValueError(f"Expected a tensor or dict in {path}, got {type(loaded).__name__}.")

    if features.ndim != 2:
        raise ValueError(f"Expected a [rows, feature_dim] tensor in {path}, got {tuple(features.shape)}.")
    return torch.nan_to_num(features.float())


def sample_rows(features, count, rng):
    if len(features) <= count:
        return features
    indices = torch.from_numpy(rng.choice(len(features), size=count, replace=False)).long()
    return features[indices]


def pca_basis(rows, components=3):
    """Mean and the top principal axes ([D, components]) of the rows."""
    mean = rows.mean(dim=0, keepdim=True)
    centered = rows - mean
    # Exact PCA: torch.pca_lowrank is randomised and made the colours change from run to run.
    # Each axis is flipped so that its largest loading is positive, which fixes the signs too.
    _, vectors = torch.linalg.eigh((centered.T @ centered).double())
    basis = vectors[:, -components:].flip(dims=[1])
    largest = basis.abs().argmax(dim=0)
    basis = basis * torch.sign(basis[largest, torch.arange(components)])
    return mean, basis.to(rows.dtype), centered


def pca_colors(features, fit_sample=12000, seed=42, clip_percentiles=(1.0, 99.0), normalize=True, invert=False):
    """RGB uint8 colours from the first three principal components of one descriptor set."""
    features = torch.nan_to_num(features)
    if normalize:
        features = torch.nn.functional.normalize(features, dim=1)

    rng = np.random.default_rng(seed)
    mean, basis, _ = pca_basis(sample_rows(features, fit_sample, rng))
    projected = ((features - mean) @ basis).cpu().numpy()

    low, high = np.percentile(projected, clip_percentiles, axis=0)
    span = np.maximum(high - low, 1e-6)
    colors = np.clip((projected - low) / span, 0.0, 1.0)
    if invert:
        colors = 1.0 - colors
    return (colors * 255).astype(np.uint8)


def shared_sample(feature_sets, count, seed, normalize=True):
    """The same rows are drawn for a given seed, so PCA and k-means see identical samples."""
    rng = np.random.default_rng(seed)
    samples = []
    for features in feature_sets:
        if normalize:
            features = torch.nn.functional.normalize(features, dim=1)
        samples.append(sample_rows(features, count, rng))
    return torch.cat(samples, dim=0)


class SharedPCA:
    """One 3-component PCA basis fitted jointly on several descriptor sets."""

    def __init__(self, feature_sets, sample_per_item=8000, seed=42, normalize=True, clip_percentiles=(1.0, 99.0)):
        self.normalize = normalize
        samples = shared_sample(feature_sets, sample_per_item, seed, normalize)
        self.mean, self.basis, centered = pca_basis(samples)
        projected = (centered @ self.basis).cpu().numpy()
        self.low, high = np.percentile(projected, clip_percentiles, axis=0)
        self.high = np.maximum(high, self.low + 1e-6)

    def prepare(self, features):
        return torch.nn.functional.normalize(features, dim=1) if self.normalize else features

    def project_prepared(self, rows):
        return ((rows - self.mean) @ self.basis).cpu().numpy()

    def project(self, features):
        return self.project_prepared(self.prepare(features))

    def colors(self, features):
        colors = np.clip((self.project(features) - self.low) / (self.high - self.low), 0.0, 1.0)
        return (colors * 255).astype(np.uint8)


def _cluster_vectors(pca, rows, cluster_on):
    if cluster_on == "pca":
        return pca.project_prepared(rows)
    return rows.cpu().numpy()


def fit_shared_kmeans(feature_sets, pca, k, sample_per_item=8000, seed=42, cluster_on="features"):
    """k-means on the full descriptors (default) or on their 3-D shared-PCA projection."""
    samples = shared_sample(feature_sets, sample_per_item, seed, pca.normalize)
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans.fit(_cluster_vectors(pca, samples, cluster_on))
    return kmeans


def predict_clusters(features, pca, kmeans, cluster_on="features"):
    return kmeans.predict(_cluster_vectors(pca, pca.prepare(features), cluster_on))


def cluster_centers_pca(pca, kmeans, cluster_on="features"):
    """k-means centroids in the shared-PCA coordinates, for plotting."""
    if cluster_on == "pca":
        return kmeans.cluster_centers_
    centers = torch.from_numpy(kmeans.cluster_centers_).float()
    return ((centers - pca.mean) @ pca.basis).cpu().numpy()


def cluster_colors(k):
    """k evenly spaced hues as float RGB tuples."""
    return [colorsys.hsv_to_rgb(i / k, 0.75, 0.90) for i in range(k)]


def cluster_palette(k):
    colors = [tuple(int(round(channel * 255)) for channel in rgb) for rgb in cluster_colors(k)]
    return np.array(colors, dtype=np.uint8)


def cluster_sizes(labels, k):
    """(cluster id, count, percent) for every cluster."""
    counts = np.bincount(labels, minlength=k)
    total = max(int(counts.sum()), 1)
    return [(i, int(counts[i]), 100.0 * counts[i] / total) for i in range(k)]
