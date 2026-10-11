from src.engine.verdict.verdict import VerdictEngine


def test_noisy_decoy_is_not_causally_implicated():
    engine = VerdictEngine()

    # Representative multi-seed C5 result.
    result = engine.classify(
        asr_all=0.9940,
        asr_without=0.9881,
        pooled_std=0.004578,
    )

    assert result["verdict"] == "CLEARED"


def test_real_backdoor_is_causally_implicated():
    engine = VerdictEngine()

    # Representative multi-seed C3 result.
    result = engine.classify(
        asr_all=0.9940,
        asr_without=0.0072,
        pooled_std=0.001178,
    )

    assert result["verdict"] == "CAUSALLY IMPLICATED"