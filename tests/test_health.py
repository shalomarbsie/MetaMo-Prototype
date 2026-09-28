import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app
from pipeline.llm_client import check_llm_health


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_shallow_check(self):
        """When check_provider=false, return 200 OK without calling LLM."""
        response = self.client.get("/health?check_provider=false")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_health_missing_api_key(self):
        """When provider API key is missing, return 503 Service Unavailable."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": ""}):
            response = self.client.get("/health")
            self.assertEqual(response.status_code, 503)
            data = response.json()
            self.assertEqual(data["status"], "unhealthy")
            self.assertEqual(data["provider"], "openai")
            self.assertIn("missing or empty", data["error"])

    def test_health_provider_auth_failure(self):
        """When provider raises an auth/connection error, return 503 Service Unavailable."""
        with patch("pipeline.llm_client.build_chat_llm") as mock_build:
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("401 Unauthorized: Invalid API key")
            mock_build.return_value = mock_llm

            with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-dummy"}):
                response = self.client.get("/health")
                self.assertEqual(response.status_code, 503)
                data = response.json()
                self.assertEqual(data["status"], "unhealthy")
                self.assertEqual(data["provider"], "openai")
                self.assertIn("401 Unauthorized", data["error"])

    def test_health_success_mocked(self):
        """When LLM ping succeeds, return 200 OK with provider and latency details."""
        with patch("pipeline.llm_client.build_chat_llm") as mock_build:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = "pong"
            mock_build.return_value = mock_llm

            with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-dummy"}):
                response = self.client.get("/health")
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["status"], "ok")
                self.assertEqual(data["provider"], "openai")
                self.assertIsNotNone(data["latency_ms"])
                self.assertIsNone(data["error"])

    def test_check_llm_health_function(self):
        """Directly verify the dictionary contract returned by check_llm_health()."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": ""}):
            result = check_llm_health(provider_name="openai")
            self.assertFalse(result["ok"])
            self.assertEqual(result["provider"], "openai")
            self.assertIn("missing or empty", result["error"])


if __name__ == "__main__":
    unittest.main()
