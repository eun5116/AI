import random
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.simple_anomaly_pipeline import train_detector


def test_training_and_prediction():
    random.seed(0)
    normal = [[random.gauss(0, 1), random.gauss(0, 1)] for _ in range(100)]
    anomalies = [[random.gauss(5, 1), random.gauss(5, 1)] for _ in range(20)]
    train = normal + anomalies
    labels = [0] * len(normal) + [1] * len(anomalies)
    model, acc = train_detector(normal, train, labels)
    preds = model.predict(train)
    pred_acc = sum(p == y for p, y in zip(preds, labels)) / len(labels)
    assert acc > 0.9
    assert abs(pred_acc - acc) < 1e-9
