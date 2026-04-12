# tests/integration/test_datasets_api.py
"""Integration tests for the /datasets/* endpoints.

Verifies dataset listing, schema serving, defaults, and random
sample generation via the REST API.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import verify_api_key


@pytest.fixture
def client():
    """TestClient with auth disabled."""
    app = create_app()
    app.dependency_overrides[verify_api_key] = lambda: None
    with TestClient(app) as tc:
        yield tc
    app.dependency_overrides.clear()


class TestListDatasets:
    def test_returns_200(self, client: TestClient):
        resp = client.get("/datasets/")
        assert resp.status_code == 200

    def test_contains_datasets_array(self, client: TestClient):
        resp = client.get("/datasets/")
        body = resp.json()
        assert "datasets" in body
        assert "count" in body
        assert isinstance(body["datasets"], list)

    def test_finds_german_credit(self, client: TestClient):
        resp = client.get("/datasets/")
        ids = [d["id"] for d in resp.json()["datasets"]]
        assert "german_credit" in ids

    def test_dataset_summary_has_expected_fields(self, client: TestClient):
        resp = client.get("/datasets/")
        ds = resp.json()["datasets"][0]
        assert "id" in ds
        assert "name" in ds
        assert "n_features" in ds
        assert "n_categorical" in ds
        assert "n_numerical" in ds
        assert "n_protected" in ds


class TestGetSchema:
    def test_returns_200_for_german_credit(self, client: TestClient):
        resp = client.get("/datasets/german_credit/schema")
        assert resp.status_code == 200

    def test_schema_has_features(self, client: TestClient):
        resp = client.get("/datasets/german_credit/schema")
        body = resp.json()
        assert "features" in body
        assert len(body["features"]) == 20

    def test_features_have_expected_fields(self, client: TestClient):
        resp = client.get("/datasets/german_credit/schema")
        feat = resp.json()["features"][0]
        assert "name" in feat
        assert "type" in feat
        assert "description" in feat
        assert "default_value" in feat

    def test_returns_404_for_unknown(self, client: TestClient):
        resp = client.get("/datasets/nonexistent_dataset/schema")
        assert resp.status_code == 404

    def test_lending_club_schema(self, client: TestClient):
        resp = client.get("/datasets/lending_club/schema")
        assert resp.status_code == 200
        assert resp.json()["id"] == "lending_club"

    def test_taiwan_credit_schema(self, client: TestClient):
        resp = client.get("/datasets/taiwan_credit/schema")
        assert resp.status_code == 200
        assert resp.json()["id"] == "taiwan_credit"


class TestGetDefaults:
    def test_returns_200(self, client: TestClient):
        resp = client.get("/datasets/german_credit/defaults")
        assert resp.status_code == 200

    def test_defaults_have_all_features(self, client: TestClient):
        resp = client.get("/datasets/german_credit/defaults")
        defaults = resp.json()["defaults"]
        assert "checking_status" in defaults
        assert "age" in defaults
        assert len(defaults) == 20

    def test_returns_404_for_unknown(self, client: TestClient):
        resp = client.get("/datasets/nonexistent/defaults")
        assert resp.status_code == 404


class TestGetRandomSample:
    def test_returns_200(self, client: TestClient):
        resp = client.get("/datasets/german_credit/random")
        assert resp.status_code == 200

    def test_sample_has_all_features(self, client: TestClient):
        resp = client.get("/datasets/german_credit/random")
        sample = resp.json()["sample"]
        assert "checking_status" in sample
        assert "age" in sample
        assert len(sample) == 20

    def test_numerical_values_within_bounds(self, client: TestClient):
        resp = client.get("/datasets/german_credit/random")
        sample = resp.json()["sample"]
        assert 18 <= sample["age"] <= 120
        assert 1 <= sample["duration"] <= 120
        assert 1 <= sample["credit_amount"] <= 100000000

    def test_two_samples_differ(self, client: TestClient):
        resp1 = client.get("/datasets/german_credit/random")
        resp2 = client.get("/datasets/german_credit/random")
        # Extremely unlikely to be identical with 20 random fields
        assert resp1.json()["sample"] != resp2.json()["sample"]
