"""Industrial deployment, lifecycle, ROI and dashboard delivery gates."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from business_analysis.roi import RoiInputs, calculate_roi  # noqa: E402
from industrial_loop.dashboard import LoopStore, create_app  # noqa: E402
from industrial_loop.events import Decision, InspectionEvent, ReasonCode  # noqa: E402
from industrial_loop.factory_simulator import FactorySimulator  # noqa: E402
from model_governance.model_lifecycle import (  # noqa: E402
    LifecycleError,
    ModelLifecycleManager,
    ModelState,
    sha256_file,
)
from model_governance.rollback_simulation import run_simulation  # noqa: E402


REQUIRED_DOCS = (
    "docs/industrial-deployment/industrial-requirement-spec.md",
    "docs/industrial-deployment/system-deployment-architecture.md",
    "docs/industrial-deployment/operation-manual.md",
    "docs/industrial-deployment/maintenance-guide.md",
    "docs/change-management.md",
    "docs/business-value-analysis.md",
    "docs/industrial-platform-maturity-report.md",
)

REQUIRED_DOC_CONTENT = (
    ("docs/industrial-deployment/industrial-requirement-spec.md", "## 2. 业务目标"),
    ("docs/industrial-deployment/industrial-requirement-spec.md", "## 3. 技术指标"),
    ("docs/industrial-deployment/industrial-requirement-spec.md", "## 4. SLA"),
    ("docs/industrial-deployment/industrial-requirement-spec.md", "## 5. 安全要求"),
    ("docs/industrial-deployment/industrial-requirement-spec.md", "## 6. 验收标准"),
    ("docs/industrial-deployment/system-deployment-architecture.md", "## 2. 数据流"),
    ("docs/industrial-deployment/system-deployment-architecture.md", "## 3. 控制流"),
    ("docs/industrial-deployment/system-deployment-architecture.md", "## 4. 网络边界"),
    ("docs/industrial-deployment/system-deployment-architecture.md", "## 5. 权限"),
    ("docs/industrial-deployment/system-deployment-architecture.md", "## 6. 故障策略"),
    ("docs/industrial-deployment/operation-manual.md", "### Camera Failure"),
    ("docs/industrial-deployment/operation-manual.md", "### AI Failure"),
    ("docs/industrial-deployment/maintenance-guide.md", "## 4. 回滚流程"),
)


def _artifact(path: Path, content: bytes = b"immutable-model-artifact") -> tuple[Path, str]:
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def _manager(tmp_path: Path) -> tuple[ModelLifecycleManager, Path, str]:
    artifact, digest = _artifact(tmp_path / "model.bin")
    manager = ModelLifecycleManager(tmp_path / "history.json", project_root=tmp_path)
    return manager, artifact, digest


def _metrics(**updates: float) -> dict[str, float]:
    metrics = {"image_auroc": 0.82, "pixel_auroc": 0.92, "aupro": 0.80}
    metrics.update(updates)
    return metrics


def _production_and_candidate(tmp_path: Path) -> tuple[ModelLifecycleManager, dict[str, str]]:
    old_artifact, old_hash = _artifact(tmp_path / "1.2.0.bin", b"previous-production")
    new_artifact, new_hash = _artifact(tmp_path / "1.3.0.bin", b"new-candidate")
    manager = ModelLifecycleManager(tmp_path / "history.json", project_root=tmp_path)
    manager.register("1.2.0", old_artifact, old_hash, operator="validator")
    manager.validate("1.2.0", metrics=_metrics(), operator="validator")
    manager.promote("1.2.0", operator="approver")
    manager.promote("1.2.0", operator="approver")
    manager.register("1.3.0", new_artifact, new_hash, operator="validator")
    manager.validate("1.3.0", metrics=_metrics(image_auroc=0.83), operator="validator")
    manager.promote("1.3.0", operator="approver")
    return manager, {"old": old_hash, "new": new_hash}


class TestDeploymentDocuments:
    @pytest.mark.parametrize("relative_path", REQUIRED_DOCS)
    def test_required_document_exists_and_is_not_empty(self, relative_path: str):
        path = ROOT / relative_path
        assert path.is_file()
        assert len(path.read_text(encoding="utf-8")) > 300

    @pytest.mark.parametrize(("relative_path", "heading"), REQUIRED_DOC_CONTENT)
    def test_required_document_content(self, relative_path: str, heading: str):
        assert heading in (ROOT / relative_path).read_text(encoding="utf-8")

    def test_roi_numbers_are_marked_as_simulation_assumptions(self):
        text = (ROOT / "docs/business-value-analysis.md").read_text(encoding="utf-8")
        assert text.count("simulation assumption") >= 4

    def test_change_management_prohibits_direct_replacement(self):
        text = (ROOT / "docs/change-management.md").read_text(encoding="utf-8")
        assert "现场直接替换模型" in text
        assert "fail closed" in text

    def test_maturity_report_has_capability_matrix_and_gaps(self):
        text = (ROOT / "docs/industrial-platform-maturity-report.md").read_text(encoding="utf-8")
        assert "工业能力矩阵" in text
        assert "与真实工业系统的差距" in text
        assert "下一步建议" in text


class TestLifecycleManager:
    @pytest.mark.parametrize("state", list(ModelState))
    def test_model_state_values_are_stable(self, state: ModelState):
        assert state.value == state.name

    def test_register_creates_development_record(self, tmp_path: Path):
        manager, artifact, digest = _manager(tmp_path)
        row = manager.register("1.0.0", artifact, digest, operator="alice")
        assert row["state"] == "DEVELOPMENT"
        assert row["artifact_hash"] == digest

    def test_duplicate_registration_is_blocked(self, tmp_path: Path):
        manager, artifact, digest = _manager(tmp_path)
        manager.register("1.0.0", artifact, digest, operator="alice")
        with pytest.raises(LifecycleError, match="MODEL_ALREADY_REGISTERED"):
            manager.register("1.0.0", artifact, digest, operator="alice")

    def test_missing_artifact_is_blocked(self, tmp_path: Path):
        manager = ModelLifecycleManager(tmp_path / "history.json", project_root=tmp_path)
        with pytest.raises(LifecycleError, match="ARTIFACT_MISSING"):
            manager.register("1.0.0", "missing.bin", "0" * 64, operator="alice")

    def test_hash_mismatch_is_blocked(self, tmp_path: Path):
        manager, artifact, _ = _manager(tmp_path)
        with pytest.raises(LifecycleError, match="ARTIFACT_HASH_MISMATCH"):
            manager.register("1.0.0", artifact, "0" * 64, operator="alice")

    def test_validation_records_metrics(self, tmp_path: Path):
        manager, artifact, digest = _manager(tmp_path)
        manager.register("1.0.0", artifact, digest, operator="alice")
        row = manager.validate("1.0.0", metrics=_metrics(), operator="bob")
        assert row["state"] == "VALIDATED"
        assert row["metrics"]["aupro"] == 0.80

    @pytest.mark.parametrize("missing", ["image_auroc", "pixel_auroc", "aupro"])
    def test_each_missing_metric_blocks_validation(self, tmp_path: Path, missing: str):
        manager, artifact, digest = _manager(tmp_path)
        manager.register("1.0.0", artifact, digest, operator="alice")
        metrics = _metrics()
        metrics.pop(missing)
        with pytest.raises(LifecycleError, match="METRIC_MISSING"):
            manager.validate("1.0.0", metrics=metrics, operator="bob")

    def test_first_promotion_creates_candidate(self, tmp_path: Path):
        manager, artifact, digest = _manager(tmp_path)
        manager.register("1.0.0", artifact, digest, operator="alice")
        manager.validate("1.0.0", metrics=_metrics(), operator="bob")
        assert manager.promote("1.0.0", operator="carol")["state"] == "CANDIDATE"

    def test_second_promotion_creates_production(self, tmp_path: Path):
        manager, artifact, digest = _manager(tmp_path)
        manager.register("1.0.0", artifact, digest, operator="alice")
        manager.validate("1.0.0", metrics=_metrics(), operator="bob")
        manager.promote("1.0.0", operator="carol")
        assert manager.promote("1.0.0", operator="carol")["state"] == "PRODUCTION"

    def test_direct_development_promotion_is_blocked(self, tmp_path: Path):
        manager, artifact, digest = _manager(tmp_path)
        manager.register("1.0.0", artifact, digest, operator="alice")
        with pytest.raises(LifecycleError, match="INVALID_TRANSITION"):
            manager.promote("1.0.0", operator="carol")

    def test_missing_metric_blocks_promotion(self, tmp_path: Path):
        manager, artifact, digest = _manager(tmp_path)
        manager.register("1.0.0", artifact, digest, operator="alice")
        manager.validate("1.0.0", metrics=_metrics(), operator="bob")
        data = json.loads(manager.history_path.read_text(encoding="utf-8"))
        data["records"][-1]["metrics"].pop("aupro")
        manager.history_path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(LifecycleError, match="METRIC_MISSING"):
            manager.promote("1.0.0", operator="carol")

    def test_artifact_tamper_blocks_promotion(self, tmp_path: Path):
        manager, artifact, digest = _manager(tmp_path)
        manager.register("1.0.0", artifact, digest, operator="alice")
        manager.validate("1.0.0", metrics=_metrics(), operator="bob")
        artifact.write_bytes(b"tampered")
        with pytest.raises(LifecycleError, match="ARTIFACT_HASH_MISMATCH"):
            manager.promote("1.0.0", operator="carol")

    def test_retire_records_transition(self, tmp_path: Path):
        manager, artifact, digest = _manager(tmp_path)
        manager.register("1.0.0", artifact, digest, operator="alice")
        row = manager.retire("1.0.0", operator="carol")
        assert row["transition"] == "DEVELOPMENT -> RETIRED"

    def test_candidate_failure_rolls_back_previous_production(self, tmp_path: Path):
        manager, hashes = _production_and_candidate(tmp_path)
        result = manager.rollback("1.3.0", "1.2.0", operator="carol", reason="inference failure")
        assert result["status"] == "COMPLETED"
        assert result["artifact_hash"] == hashes["old"]
        assert manager.operations_snapshot()["current_model_version"] == "1.2.0"

    def test_production_failure_restores_retired_version(self, tmp_path: Path):
        manager, hashes = _production_and_candidate(tmp_path)
        manager.promote("1.3.0", operator="carol")
        result = manager.rollback("1.3.0", "1.2.0", operator="carol", reason="quality gate failure")
        assert result["artifact_hash"] == hashes["old"]
        states = {row["model_version"]: row["state"] for row in manager.model_snapshot()["versions"]}
        assert states == {"1.2.0": "PRODUCTION", "1.3.0": "RETIRED"}

    def test_invalid_rollback_target_is_blocked(self, tmp_path: Path):
        manager, _ = _production_and_candidate(tmp_path)
        with pytest.raises(LifecycleError, match="MODEL_NOT_REGISTERED"):
            manager.rollback("1.3.0", "0.9.0", operator="carol", reason="failure")

    def test_operations_snapshot_exposes_governance_fields(self, tmp_path: Path):
        manager, hashes = _production_and_candidate(tmp_path)
        snapshot = manager.operations_snapshot()
        assert snapshot == {
            "available": True,
            "current_model_version": "1.2.0",
            "lifecycle_state": "PRODUCTION",
            "artifact_hash": hashes["old"],
            "rollback_status": None,
        }

    def test_model_snapshot_exposes_metrics_and_approval(self, tmp_path: Path):
        manager, _ = _production_and_candidate(tmp_path)
        snapshot = manager.model_snapshot()
        assert snapshot["available"] is True
        assert all({"model_version", "metrics", "approval_status"} <= set(row) for row in snapshot["versions"])

    def test_unreadable_history_fails_closed(self, tmp_path: Path):
        path = tmp_path / "history.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(LifecycleError, match="MODEL_HISTORY_UNREADABLE"):
            ModelLifecycleManager(path)

    def test_wrong_history_schema_fails_closed(self, tmp_path: Path):
        path = tmp_path / "history.json"
        path.write_text('{"schema_version":"wrong","records":[]}', encoding="utf-8")
        with pytest.raises(LifecycleError, match="MODEL_HISTORY_SCHEMA_MISMATCH"):
            ModelLifecycleManager(path)

    def test_seed_history_matches_frozen_candidate_manifest(self):
        history = json.loads((ROOT / "model_governance/model_history.json").read_text(encoding="utf-8"))
        record = history["records"][0]
        assert sha256_file(ROOT / record["artifact_path"]) == record["artifact_hash"]
        assert record["state"] == "CANDIDATE"

    def test_runnable_rollback_simulation_restores_identical_artifact(self, tmp_path: Path):
        report = run_simulation(tmp_path)
        assert report["inference_outcome"] == "FAILURE"
        assert report["rollback_status"] == "COMPLETED"
        assert report["restored_version"] == "1.2.0"
        assert report["artifact_hash_consistent"] is True


ROI_INPUTS = RoiInputs(
    inspectors=6,
    annual_cost_per_inspector=120_000,
    automation_rate=0.70,
    annual_ai_operating_cost=180_000,
    initial_investment=1_350_000,
    baseline_escape_rate=0.0025,
    ai_escape_rate=0.001,
    annual_units=1_200_000,
    escape_cost_per_unit=180,
)


class TestRoiCalculation:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("labor_baseline", 720_000),
            ("labor_saving", 504_000),
            ("escape_avoidance", 324_000),
            ("annual_net_benefit", 648_000),
            ("roi", 0.48),
            ("payback_years", 1_350_000 / 648_000),
            ("detection_improvement", 0.0015),
        ],
    )
    def test_business_value_formula(self, key: str, expected: float):
        assert calculate_roi(ROI_INPUTS)[key] == pytest.approx(expected)

    @pytest.mark.parametrize(
        "updates",
        [
            {"inspectors": -1},
            {"annual_units": -1},
            {"annual_cost_per_inspector": -1},
            {"automation_rate": 1.1},
            {"baseline_escape_rate": -0.1},
            {"ai_escape_rate": 1.1},
        ],
    )
    def test_invalid_roi_inputs_are_rejected(self, updates: dict):
        values = {**ROI_INPUTS.__dict__, **updates}
        with pytest.raises(ValueError):
            calculate_roi(RoiInputs(**values))

    def test_zero_investment_has_no_ratio(self):
        values = {**ROI_INPUTS.__dict__, "initial_investment": 0}
        assert calculate_roi(RoiInputs(**values))["roi"] is None

    def test_non_positive_benefit_has_no_payback(self):
        values = {**ROI_INPUTS.__dict__, "annual_ai_operating_cost": 2_000_000}
        assert calculate_roi(RoiInputs(**values))["payback_years"] is None


class TestGovernanceDashboard:
    def test_unconfigured_operations_api_fails_closed(self):
        body = TestClient(create_app(LoopStore())).get("/api/operations").json()
        assert body["available"] is False
        assert body["artifact_hash"] is None

    def test_unconfigured_model_api_returns_no_versions(self):
        body = TestClient(create_app(LoopStore())).get("/api/model").json()
        assert body == {"available": False, "versions": [], "history": []}

    @pytest.mark.parametrize("path", ["/", "/operations", "/model"])
    def test_dashboard_pages_are_served(self, path: str):
        response = TestClient(create_app(LoopStore())).get(path)
        assert response.status_code == 200
        assert "Operations" in response.text
        assert "Version history and approvals" in response.text

    def test_configured_operations_api_returns_current_production(self, tmp_path: Path):
        manager, hashes = _production_and_candidate(tmp_path)
        body = TestClient(create_app(LoopStore(), lifecycle_manager=manager)).get("/api/operations").json()
        assert body["current_model_version"] == "1.2.0"
        assert body["artifact_hash"] == hashes["old"]

    def test_configured_model_api_returns_history_metrics_and_approval(self, tmp_path: Path):
        manager, _ = _production_and_candidate(tmp_path)
        body = TestClient(create_app(LoopStore(), lifecycle_manager=manager)).get("/api/model").json()
        assert len(body["versions"]) == 2
        assert len(body["history"]) >= 7
        assert all(row["metrics"] for row in body["versions"])

    def test_dashboard_html_calls_both_governance_apis(self):
        page = TestClient(create_app(LoopStore())).get("/").text
        assert 'fetch("/api/operations")' in page
        assert 'fetch("/api/model")' in page

    def test_factory_dashboard_loads_frozen_candidate_governance(self):
        simulator = FactorySimulator(products=1, seed=1)
        body = TestClient(simulator.app).get("/api/operations").json()
        assert body["available"] is True
        assert body["current_model_version"] == "1.3.0"
        assert body["lifecycle_state"] == "CANDIDATE"
        assert body["artifact_hash"] == "284e7e7c6aa57158088e833b10c3f3c8c3156d94149323eaf50d87343028d909"

    def test_lifecycle_inference_failure_hold_and_rollback_e2e(self, tmp_path: Path):
        manager, hashes = _production_and_candidate(tmp_path)
        store = LoopStore()
        store.add_event(
            InspectionEvent(
                product_id="STEEL-FAIL-001",
                batch_id="BATCH-001",
                camera_id="CAM-01",
                model_version="1.3.0",
                artifact_version="candidate",
                decision=Decision.HOLD,
                reason_code=ReasonCode.AI_SYSTEM_FAILURE,
                error_detail="simulated inference failure",
            )
        )
        rollback = manager.rollback("1.3.0", "1.2.0", operator="release-operator", reason="inference failure")
        client = TestClient(create_app(store, lifecycle_manager=manager))
        assert client.get("/api/events").json()[0]["decision"] == "HOLD"
        operations = client.get("/api/operations").json()
        assert operations["current_model_version"] == "1.2.0"
        assert operations["artifact_hash"] == hashes["old"]
        assert rollback["status"] == "COMPLETED"
