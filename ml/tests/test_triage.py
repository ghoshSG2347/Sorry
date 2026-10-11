import numpy as np
import torch

from src.engine.triage.triage import TriageEngine


class DummyModel:
    def eval(self):
        return self

    def __call__(self, images):
        # 10-class dummy logits.
        return torch.zeros(
            (images.shape[0], 10),
            dtype=torch.float32,
        )

    def extract_features(self, images):
        # Deterministic low-dimensional representation.
        flattened = images.view(images.shape[0], -1)
        return flattened[:, :8]


def test_triage_identifies_all_contributors():
    engine = TriageEngine(
        model=DummyModel()
    )

    results = engine.run(
        contributors=(1, 2, 3, 4, 5)
    )

    assert len(results) == 5

    contributors = {
        result["contributor"]
        for result in results
    }

    assert contributors == {1, 2, 3, 4, 5}


def test_triage_contains_required_signals():
    engine = TriageEngine(
        model=DummyModel()
    )

    results = engine.run(
        contributors=(1, 2, 3, 4, 5)
    )

    required_fields = {
        "contributor",
        "mean_loss",
        "std_loss",
        "minority_fraction",
        "loss_rank",
        "cluster_rank",
        "combined_rank_score",
        "normalized_loss_score",
        "normalized_cluster_score",
        "normalized_suspicion_score",
    }

    for result in results:
        assert required_fields.issubset(
            result.keys()
        )


def test_triage_ranking_is_deterministic():
    engine1 = TriageEngine(
        model=DummyModel(),
        random_state=42,
    )

    engine2 = TriageEngine(
        model=DummyModel(),
        random_state=42,
    )

    results1 = engine1.run()
    results2 = engine2.run()

    ranking1 = [
        result["contributor"]
        for result in results1
    ]

    ranking2 = [
        result["contributor"]
        for result in results2
    ]

    assert ranking1 == ranking2