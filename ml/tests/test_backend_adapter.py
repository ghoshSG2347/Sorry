import json
from pathlib import Path
import pytest

from handoff.backend.app.services.ml_runner import (
    MLRunner,
    MLRunnerError,
    MLScenarioNotSupportedError,
    MLInvalidConfigurationError,
    run_scenario,
    SUPPORTED_SCENARIOS,
    SUPPORTED_MODES,
)


@pytest.fixture
def ml_root():
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(ml_root):
    return MLRunner(ml_root=ml_root)


@pytest.fixture
def schema(ml_root):
    schema_path = ml_root / "schema" / "ml_result_schema.json"
    assert schema_path.is_file(), f"Schema file not found at {schema_path}"
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_adapter_initialization(runner, ml_root):
    assert runner.ml_root == ml_root
    assert runner.replay_dir.is_dir()
    assert runner.protocol_path.is_file()


def test_supported_scenarios_and_modes(runner):
    scenarios = runner.get_supported_scenarios()
    modes = runner.get_supported_modes()

    assert "S1_BACKDOOR_C3" in scenarios
    assert "S2_NOISY_DECOY_C5" in scenarios
    assert "S1_S2_COMBINED" in scenarios
    assert "S3_COLLUSION_C2_C3" in scenarios

    assert "replay" in modes
    assert "live" in modes


def test_protocol_loading(runner):
    protocol = runner.load_protocol()
    assert protocol["protocol_id"] == "POISONTRACE_PROTOCOL_V1"
    assert protocol["protocol_version"] == "1.0.0"
    assert protocol["dataset"]["name"] == "Fashion-MNIST"
    assert protocol["model"]["parameter_count"] == 1701130
    assert protocol["attack"]["target_class"] == 0
    assert protocol["individual_attribution_rules"]["causal_threshold_multiplier"] == 0.50
    assert protocol["individual_attribution_rules"]["max_clean_accuracy_drop"] == 0.02
    assert protocol["collusion_attribution_rules"]["group"] == [2, 3]


def test_replay_s1_s2_combined_returns_valid_fixture(runner):
    result = runner.run_scenario(
        scenario_id="S1_S2_COMBINED",
        mode="replay",
        run_id="test-run-s1-s2-001",
    )

    assert result["schema_version"] == "1.0.0"
    assert result["run_id"] == "test-run-s1-s2-001"
    assert result["scenario_id"] == "S1_S2_COMBINED"
    assert result["mode"] == "replay"
    assert result["status"] == "SUCCESS"
    assert result["execution_time_seconds"] == 0.0

    # Baseline metrics
    assert round(result["baseline"]["clean_accuracy"]["mean"], 4) == 0.9047
    assert round(result["baseline"]["asr"]["mean"], 4) == 0.9940

    # Attacker C3 must be CAUSALLY IMPLICATED
    c3 = result["contributors"]["C3"]
    assert c3["verdict"] == "CAUSALLY IMPLICATED"
    assert round(c3["delta_asr"]["mean"], 4) == 0.9868
    assert round(c3["clean_accuracy_drop"]["mean"], 4) == 0.0086

    # C1, C2, C4, C5 must be CLEARED
    for cid in ["C1", "C2", "C4", "C5"]:
        assert result["contributors"][cid]["verdict"] == "CLEARED"


def test_replay_s3_collusion_returns_valid_fixture(runner):
    result = runner.run_scenario(
        scenario_id="S3_COLLUSION_C2_C3",
        mode="replay",
        run_id="test-run-s3-001",
    )

    assert result["schema_version"] == "1.0.0"
    assert result["run_id"] == "test-run-s3-001"
    assert result["scenario_id"] == "S3_COLLUSION_C2_C3"
    assert result["mode"] == "replay"
    assert result["status"] == "SUCCESS"

    collusion = result["collusion_group"]
    assert collusion["group"] == [2, 3]
    assert collusion["individual_C2_removal_preserves_backdoor"] is True
    assert collusion["individual_C3_removal_preserves_backdoor"] is True
    assert collusion["joint_removal_removes_backdoor"] is True
    assert collusion["verdict"] == "COLLUSIVE GROUP IMPLICATED"
    assert round(collusion["delta_asr_without_group"]["mean"], 4) == 0.9799


def test_unsupported_scenario_rejected(runner):
    # Non-raising mode returns status="FAILED"
    failed_result = runner.run_scenario(
        scenario_id="UNKNOWN_SCENARIO_XYZ",
        mode="replay",
        raise_on_failure=False,
    )
    assert failed_result["status"] == "FAILED"
    assert "Unsupported scenario_id" in failed_result["error_message"]

    # Raising mode raises MLScenarioNotSupportedError
    with pytest.raises(MLScenarioNotSupportedError):
        runner.run_scenario(
            scenario_id="UNKNOWN_SCENARIO_XYZ",
            mode="replay",
            raise_on_failure=True,
        )


def test_unsupported_mode_rejected(runner):
    failed_result = runner.run_scenario(
        scenario_id="S1_S2_COMBINED",
        mode="quantum_accelerated",
        raise_on_failure=False,
    )
    assert failed_result["status"] == "FAILED"
    assert "Unsupported mode" in failed_result["error_message"]

    with pytest.raises(MLInvalidConfigurationError):
        runner.run_scenario(
            scenario_id="S1_S2_COMBINED",
            mode="quantum_accelerated",
            raise_on_failure=True,
        )


def test_failed_execution_does_not_masquerade_as_success(runner):
    # Simulate a missing fixture or failure
    broken_runner = MLRunner(ml_root=Path("/nonexistent/directory"))
    result = broken_runner.run_scenario(
        scenario_id="S1_S2_COMBINED",
        mode="replay",
        raise_on_failure=False,
    )

    assert result["status"] == "FAILED"
    assert result["error_message"] is not None
    assert result["status"] != "SUCCESS"


def test_live_and_replay_distinguishable(runner):
    replay_result = runner.run_scenario(
        scenario_id="S1_S2_COMBINED",
        mode="replay",
    )
    assert replay_result["mode"] == "replay"
    assert replay_result["execution_time_seconds"] == 0.0


def test_no_internal_server_paths_leaked(runner):
    result = runner.run_scenario(
        scenario_id="S1_S2_COMBINED",
        mode="replay",
    )
    raw_json = json.dumps(result)
    assert "C:\\Users\\" not in raw_json
    assert "/Users/" not in raw_json
    assert "OneDrive" not in raw_json


def test_output_structural_conformance(runner):
    result = runner.run_scenario(
        scenario_id="S1_S2_COMBINED",
        mode="replay",
    )

    required_fields = [
        "schema_version",
        "run_id",
        "scenario_id",
        "mode",
        "status",
        "seeds",
        "protocol_ref",
        "baseline",
        "artifact_hashes",
        "limitations",
    ]
    for field in required_fields:
        assert field in result, f"Missing required top-level field: {field}"

    assert result["protocol_ref"]["protocol_id"] == "POISONTRACE_PROTOCOL_V1"
    assert len(result["protocol_ref"]["protocol_hash"]) == 64
    assert len(result["artifact_hashes"]) > 0


def test_schema_validation_s1_s2_fixture(runner, schema):
    import jsonschema

    result = runner.run_scenario(
        scenario_id="S1_S2_COMBINED",
        mode="replay",
    )
    # Must not raise jsonschema.ValidationError
    jsonschema.validate(instance=result, schema=schema)


def test_schema_validation_s3_fixture(runner, schema):
    import jsonschema

    result = runner.run_scenario(
        scenario_id="S3_COLLUSION_C2_C3",
        mode="replay",
    )
    # Must not raise jsonschema.ValidationError
    jsonschema.validate(instance=result, schema=schema)

