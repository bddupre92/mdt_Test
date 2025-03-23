"""
Test suite for Phase 1 components.
"""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import jwt
import os
import logging
from unittest.mock import Mock, patch

from app.core.security.oauth import OAuth2Handler, Token
from app.core.security.api_key import APIKeyManager
from app.core.middleware.rate_limiter import RateLimiter
from app.core.middleware.validation import RequestValidator
from app.core.middleware.audit import AuditLogger
from app.core.monitoring.metrics import MetricsCollector

# Setup logging
logging.basicConfig(level=logging.INFO)

@pytest.fixture
def oauth_handler():
    return OAuth2Handler()

@pytest.fixture
def api_key_manager():
    return APIKeyManager()

@pytest.fixture
def rate_limiter():
    return RateLimiter()

@pytest.fixture
def request_validator():
    return RequestValidator()

@pytest.fixture
def audit_logger():
    return AuditLogger()

@pytest.fixture
def metrics_collector():
    return MetricsCollector()

class TestOAuth2:
    def test_create_access_token(self, oauth_handler):
        data = {"sub": "test_user", "scopes": ["read:predictions"]}
        token = oauth_handler.create_access_token(data)
        
        assert isinstance(token, Token)
        assert token.token_type == "bearer"
        assert isinstance(token.access_token, str)
        
        # Verify token contents
        decoded = jwt.decode(
            token.access_token,
            oauth_handler.secret_key,
            algorithms=[oauth_handler.algorithm]
        )
        assert decoded["sub"] == "test_user"
        assert decoded["scopes"] == ["read:predictions"]

class TestAPIKey:
    def test_generate_key(self, api_key_manager):
        key = api_key_manager.generate_key()
        assert key.startswith("mdt_")
        assert len(key) > 20

class TestRateLimiter:
    async def test_rate_limiting(self, rate_limiter):
        # Mock request
        request = Mock()
        request.headers = {}
        request.client.host = "127.0.0.1"
        request.url.path = "/predict"
        
        # First request should pass
        is_limited, _ = await rate_limiter.is_rate_limited(request)
        assert not is_limited
        
        # Simulate many requests
        for _ in range(101):  # Exceed limit
            await rate_limiter.is_rate_limited(request)
            
        # Next request should be limited
        is_limited, retry_after = await rate_limiter.is_rate_limited(request)
        assert is_limited
        assert retry_after is not None

class TestRequestValidator:
    async def test_validate_feature_values(self, request_validator):
        # Valid data
        valid_data = {
            "features": {
                "sleep_hours": 7.5,
                "stress_level": 5,
                "temperature": 20.5,
                "pressure": 1015.0
            }
        }
        await request_validator.validate_feature_values(valid_data)
        
        # Invalid data
        invalid_data = {
            "features": {
                "sleep_hours": 25,  # Invalid value
                "stress_level": 11  # Invalid value
            }
        }
        with pytest.raises(Exception):
            await request_validator.validate_feature_values(invalid_data)

class TestAuditLogger:
    async def test_audit_logging(self, audit_logger):
        # Mock request
        request = Mock()
        request.method = "POST"
        request.url = "http://test/predict"
        request.client.host = "127.0.0.1"
        request.headers = {"user-agent": "test-client"}
        
        # Test request logging
        request_id = await audit_logger.log_request(request)
        assert isinstance(request_id, str)
        
        # Test response logging
        await audit_logger.log_response(request_id, 200, 0.1)
        
        # Test error logging
        error = ValueError("test error")
        await audit_logger.log_error(request_id, error, 500)

class TestMetrics:
    def test_metrics_recording(self, metrics_collector):
        # Record request
        metrics_collector.record_request(
            method="POST",
            endpoint="/predict",
            status=200,
            duration=0.1
        )
        
        # Record prediction
        metrics_collector.record_prediction(
            duration=0.05,
            drift_score=0.1,
            accuracy=0.95
        )
        
        # Update resource usage
        metrics_collector.update_resource_usage(
            memory_bytes=1024*1024,  # 1MB
            cpu_seconds=1.5
        )

def test_docker_compose():
    """Test if docker-compose.yml is valid."""
    import yaml
    
    compose_path = os.path.join(
        os.path.dirname(__file__),
        "../infrastructure/docker/docker-compose.yml"
    )
    
    with open(compose_path) as f:
        compose_data = yaml.safe_load(f)
        
    assert "services" in compose_data
    required_services = ["api", "ml", "db", "redis", "prometheus", "grafana"]
    for service in required_services:
        assert service in compose_data["services"]

def test_kubernetes_configs():
    """Test if Kubernetes configs are valid."""
    import yaml
    
    base_path = os.path.join(
        os.path.dirname(__file__),
        "../infrastructure/kubernetes/base"
    )
    
    # Test deployment.yaml
    with open(os.path.join(base_path, "deployment.yaml")) as f:
        deploy_data = yaml.safe_load(f)
        assert deploy_data["kind"] == "Deployment"
        assert deploy_data["spec"]["replicas"] == 3
    
    # Test service.yaml
    with open(os.path.join(base_path, "service.yaml")) as f:
        svc_data = yaml.safe_load(f)
        assert svc_data["kind"] == "Service"
        assert svc_data["spec"]["type"] == "ClusterIP"
    
    # Test ingress.yaml
    with open(os.path.join(base_path, "ingress.yaml")) as f:
        ing_data = yaml.safe_load(f)
        assert ing_data["kind"] == "Ingress"
        assert "tls" in ing_data["spec"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
