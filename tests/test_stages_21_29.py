"""Tests for Stages 21-29.

Stage 21 — CI workflows exist (file presence test)
Stage 22 — docs/ and mkdocs.yml exist
Stage 23 — benchmarks/ scripts exist
Stage 24 — query_strategy="gp"
Stage 25 — cost parameter in reward()
Stage 26 — policy="ensemble"
Stage 27 — GET /metrics in server
Stage 28 — k8s manifests exist
Stage 29 — ModelRegistry
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Stage 21 — CI/CD pipeline
# ---------------------------------------------------------------------------

class TestStage21CI:
    def test_ci_workflow_exists(self):
        assert (REPO_ROOT / ".github" / "workflows" / "ci.yml").exists()

    def test_publish_workflow_exists(self):
        assert (REPO_ROOT / ".github" / "workflows" / "publish.yml").exists()

    def test_ci_yml_contains_matrix(self):
        content = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "3.10" in content
        assert "3.11" in content
        assert "3.12" in content

    def test_publish_yml_contains_oidc(self):
        content = (REPO_ROOT / ".github" / "workflows" / "publish.yml").read_text()
        assert "id-token" in content
        assert "pypi-publish" in content


# ---------------------------------------------------------------------------
# Stage 22 — Documentation site
# ---------------------------------------------------------------------------

class TestStage22Docs:
    def test_mkdocs_yml_exists(self):
        assert (REPO_ROOT / "mkdocs.yml").exists()

    def test_docs_index_exists(self):
        assert (REPO_ROOT / "docs" / "index.md").exists()

    def test_docs_getting_started_exists(self):
        assert (REPO_ROOT / "docs" / "getting_started.md").exists()

    def test_docs_physics_exists(self):
        assert (REPO_ROOT / "docs" / "physics.md").exists()

    def test_docs_api_reference_exists(self):
        assert (REPO_ROOT / "docs" / "api_reference.md").exists()

    def test_docs_workflow_exists(self):
        assert (REPO_ROOT / ".github" / "workflows" / "docs.yml").exists()


# ---------------------------------------------------------------------------
# Stage 23 — Benchmarks
# ---------------------------------------------------------------------------

class TestStage23Benchmarks:
    def test_benchmark_runner_exists(self):
        assert (REPO_ROOT / "benchmarks" / "run_benchmarks.py").exists()

    def test_plot_benchmarks_exists(self):
        assert (REPO_ROOT / "benchmarks" / "plot_benchmarks.py").exists()

    def test_results_directory_exists(self):
        assert (REPO_ROOT / "benchmarks" / "results").is_dir()

    def test_summary_csv_exists(self):
        assert (REPO_ROOT / "benchmarks" / "results" / "summary.csv").exists()

    def test_readme_has_results_table(self):
        readme = (REPO_ROOT / "README.md").read_text()
        assert "Benchmark Results" in readme
        assert "iris" in readme


# ---------------------------------------------------------------------------
# Stage 24 — GP uncertainty query_strategy="gp"
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_clf_data():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 4))
    y = (X[:, 0] > 0).astype(int)
    return X, y


class TestStage24GP:
    def test_gp_strategy_accepted(self, small_clf_data):
        from physml import myco
        X, y = small_clf_data
        agent = myco(query_strategy="gp")
        agent.fit(X[:40], y[:40])
        # Should return a valid index (not raise)
        idx = agent.select_informative(X[40:])
        assert 0 <= idx < 40

    def test_gp_strategy_fallback_too_few_labels(self, small_clf_data):
        """With < 3 labels the GP falls back gracefully."""
        from physml.agent import PhysicsAgent
        from physml.estimator import PhysicsPredictor

        X, y = small_clf_data
        pred = PhysicsPredictor(backend="neural", n_cycles=5)
        pred.fit(X[:40], y[:40])
        agent = PhysicsAgent(pred, query_strategy="gp")
        # No GP data yet — _gp_select returns None → falls back
        result = agent._gp_select(X[40:60])
        assert result is None or isinstance(result, int)

    def test_gp_select_after_rewards(self, small_clf_data):
        from physml import myco
        X, y = small_clf_data
        agent = myco(query_strategy="gp")
        agent.fit(X[:20], y[:20])
        # Add some rewards to populate GP training data
        for i in range(5):
            agent.reward(X[20 + i], y[20 + i])
        idx = agent.select_informative(X[30:50])
        assert 0 <= idx < 20


# ---------------------------------------------------------------------------
# Stage 25 — Cost-aware oracle
# ---------------------------------------------------------------------------

class TestStage25Cost:
    def test_cost_parameter_accepted(self, small_clf_data):
        from physml import myco
        X, y = small_clf_data
        agent = myco(policy="bandit")
        agent.fit(X[:40], y[:40])
        # Should not raise
        agent.reward(X[40], y[40], cost=0.5)
        agent.reward(X[41], y[41], cost=2.0)

    def test_total_oracle_cost_tracked(self, small_clf_data):
        from physml import myco
        X, y = small_clf_data
        agent = myco()
        agent.fit(X[:40], y[:40])
        agent.reward(X[40], y[40], cost=3.0)
        agent.reward(X[41], y[41], cost=1.5)
        r = agent.report()
        assert r["agent"]["total_oracle_cost"] == pytest.approx(4.5)

    def test_default_cost_is_one(self, small_clf_data):
        from physml import myco
        X, y = small_clf_data
        agent = myco()
        agent.fit(X[:40], y[:40])
        agent.reward(X[40], y[40])  # no cost kwarg
        r = agent.report()
        assert r["agent"]["total_oracle_cost"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Stage 26 — Ensemble / query-by-committee
# ---------------------------------------------------------------------------

class TestStage26Ensemble:
    def test_ensemble_policy_accepted(self, small_clf_data):
        from physml import myco
        X, y = small_clf_data
        agent = myco(policy="ensemble")
        agent.fit(X[:40], y[:40])
        action = agent.observe(X[40])
        assert action.action in ("predict", "ask", "abstain")

    def test_ensemble_disagreement_range(self, small_clf_data):
        from physml.agent import PhysicsAgent
        from physml.estimator import PhysicsPredictor

        X, y = small_clf_data
        pred = PhysicsPredictor(backend="neural", n_cycles=5)
        pred.fit(X[:40], y[:40])
        agent = PhysicsAgent(pred, policy="ensemble")
        # Populate GP/ensemble data
        agent._gp_X = [X[:40]]
        agent._gp_y = [y[:40]]
        d = agent._ensemble_disagreement(X[40:50])
        assert 0.0 <= d <= 1.0

    def test_ensemble_policy_report_has_policy_key(self, small_clf_data):
        from physml import myco
        X, y = small_clf_data
        agent = myco(policy="ensemble")
        agent.fit(X[:40], y[:40])
        r = agent.report()
        assert r["policy"] == "ensemble"


# ---------------------------------------------------------------------------
# Stage 27 — /metrics endpoint
# ---------------------------------------------------------------------------

class TestStage27Metrics:
    def test_metrics_endpoint_in_server_source(self):
        server_src = (REPO_ROOT / "physml" / "server.py").read_text()
        assert "/metrics" in server_src
        assert "physml_n_observations_total" in server_src

    @pytest.mark.skipif(
        not (
            __import__("importlib").util.find_spec("fastapi") is not None
            and __import__("importlib").util.find_spec("httpx") is not None
        ),
        reason="fastapi and httpx are required",
    )
    def test_metrics_returns_prometheus_text(self):
        from fastapi.testclient import TestClient
        from physml.server import create_app

        client = TestClient(create_app())
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "physml_n_observations_total" in resp.text
        assert "physml_oracle_calls_total" in resp.text


# ---------------------------------------------------------------------------
# Stage 28 — Kubernetes manifests
# ---------------------------------------------------------------------------

class TestStage28Kubernetes:
    def test_deployment_yaml_exists(self):
        assert (REPO_ROOT / "k8s" / "deployment.yaml").exists()

    def test_service_yaml_exists(self):
        assert (REPO_ROOT / "k8s" / "service.yaml").exists()

    def test_hpa_yaml_exists(self):
        assert (REPO_ROOT / "k8s" / "hpa.yaml").exists()

    def test_deployment_references_image(self):
        content = (REPO_ROOT / "k8s" / "deployment.yaml").read_text()
        assert "image:" in content

    def test_hpa_references_deployment(self):
        content = (REPO_ROOT / "k8s" / "hpa.yaml").read_text()
        assert "HorizontalPodAutoscaler" in content
        assert "maxReplicas" in content


# ---------------------------------------------------------------------------
# Stage 29 — ModelRegistry
# ---------------------------------------------------------------------------

class TestStage29Registry:
    def test_registry_importable(self):
        from physml.registry import ModelRegistry  # noqa: F401

    def test_registry_exported_from_physml(self):
        from physml import ModelRegistry  # noqa: F401

    def test_log_returns_run_id(self, tmp_path, small_clf_data):
        from physml import myco, ModelRegistry
        X, y = small_clf_data
        agent = myco()
        agent.fit(X[:40], y[:40])
        reg = ModelRegistry(tmp_path / "runs.jsonl")
        run_id = reg.log(agent, X[:40], y[:40], tags={"test": True}, save_agent=True)
        assert isinstance(run_id, str) and len(run_id) == 32

    def test_list_runs_returns_records(self, tmp_path, small_clf_data):
        from physml import myco, ModelRegistry
        X, y = small_clf_data
        agent = myco()
        agent.fit(X[:40], y[:40])
        reg = ModelRegistry(tmp_path / "runs.jsonl")
        reg.log(agent, X[:40], y[:40], save_agent=False)
        runs = reg.list_runs()
        # Can be DataFrame or list
        assert len(runs) >= 1

    def test_get_run_by_id(self, tmp_path, small_clf_data):
        from physml import myco, ModelRegistry
        X, y = small_clf_data
        agent = myco()
        agent.fit(X[:40], y[:40])
        reg = ModelRegistry(tmp_path / "runs.jsonl")
        run_id = reg.log(agent, X[:40], y[:40], save_agent=False)
        record = reg.get_run(run_id)
        assert record["run_id"] == run_id
        assert "dataset_hash" in record
        assert "n_samples" in record

    def test_load_agent(self, tmp_path, small_clf_data):
        from physml import myco, ModelRegistry
        X, y = small_clf_data
        agent = myco()
        agent.fit(X[:40], y[:40])
        reg = ModelRegistry(tmp_path / "runs.jsonl")
        run_id = reg.log(agent, X[:40], y[:40], save_agent=True)
        loaded = reg.load_agent(run_id)
        assert loaded is not None

    def test_delete_run(self, tmp_path, small_clf_data):
        from physml import myco, ModelRegistry
        X, y = small_clf_data
        agent = myco()
        agent.fit(X[:40], y[:40])
        reg = ModelRegistry(tmp_path / "runs.jsonl")
        run_id = reg.log(agent, X[:40], y[:40], save_agent=False)
        reg.delete_run(run_id)
        with pytest.raises(KeyError):
            reg.get_run(run_id)

    def test_unknown_run_raises(self, tmp_path):
        from physml.registry import ModelRegistry
        reg = ModelRegistry(tmp_path / "runs.jsonl")
        with pytest.raises(KeyError):
            reg.get_run("nonexistent_id")

    def test_registry_repr(self, tmp_path):
        from physml.registry import ModelRegistry
        reg = ModelRegistry(tmp_path / "empty.jsonl")
        r = repr(reg)
        assert "ModelRegistry" in r
        assert "n_runs=0" in r
