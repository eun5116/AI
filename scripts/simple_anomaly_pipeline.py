"""Simple anomaly detection pipeline without external dependencies.

This module demonstrates a two-stage anomaly detection approach:
1. Unsupervised: estimate normal data distribution to compute anomaly scores.
2. Supervised: determine a score threshold using labeled data to measure accuracy.

The implementation uses only the Python standard library so it can run in
minimal environments.
"""

from typing import List, Sequence, Tuple


def _mean_variance(data: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    """Return per-dimension mean and variance from normal data."""
    if not data:
        raise ValueError("normal data is empty")
    dim = len(data[0])
    mean = [0.0] * dim
    for row in data:
        for i, v in enumerate(row):
            mean[i] += v
    n = float(len(data))
    mean = [m / n for m in mean]
    var = [0.0] * dim
    for row in data:
        for i, v in enumerate(row):
            d = v - mean[i]
            var[i] += d * d
    var = [v / n for v in var]
    return mean, var


def anomaly_score(sample: Sequence[float], mean: Sequence[float], var: Sequence[float]) -> float:
    """Compute normalized squared distance of a sample from the mean."""
    return sum(((x - m) ** 2) / (v + 1e-12) for x, m, v in zip(sample, mean, var))


def find_threshold(scores: Sequence[float], labels: Sequence[int]) -> Tuple[float, float]:
    """Find score threshold with best accuracy on labeled data.

    Returns (threshold, accuracy).
    """
    paired = sorted(zip(scores, labels))
    candidates = [-float("inf")] + [
        (paired[i][0] + paired[i + 1][0]) / 2.0 for i in range(len(paired) - 1)
    ] + [float("inf")]
    best_acc, best_thr = -1.0, 0.0
    total = len(paired)
    for thr in candidates:
        correct = sum((s <= thr) == (y == 0) for s, y in paired)
        acc = correct / total
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_thr, best_acc


class AnomalyDetector:
    """Detector that uses distance-based scoring with a learned threshold."""

    def __init__(self, mean: Sequence[float], var: Sequence[float], threshold: float):
        self.mean = list(mean)
        self.var = list(var)
        self.threshold = threshold

    def score(self, samples: Sequence[Sequence[float]]) -> List[float]:
        return [anomaly_score(s, self.mean, self.var) for s in samples]

    def predict(self, samples: Sequence[Sequence[float]]) -> List[int]:
        return [1 if s > self.threshold else 0 for s in self.score(samples)]


def train_detector(
    normal_data: Sequence[Sequence[float]],
    labeled_data: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> Tuple[AnomalyDetector, float]:
    """Train detector using normal data and labeled examples.

    Returns the fitted detector and the training accuracy.
    """
    mean, var = _mean_variance(normal_data)
    scores = [anomaly_score(x, mean, var) for x in labeled_data]
    threshold, acc = find_threshold(scores, labels)
    model = AnomalyDetector(mean, var, threshold)
    return model, acc
