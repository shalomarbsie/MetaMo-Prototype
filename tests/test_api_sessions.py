from __future__ import annotations

import time
from threading import Thread
from cachetools import TTLCache
from fastapi.testclient import TestClient

import api.main as main_module
from api.main import app, _states, _lock, SESSION_MAX_ENTRIES, SESSION_TTL_SECONDS
from core.state import init_state


def test_default_cache_configuration():
    """Verify that default cache configurations and type are correct."""
    assert isinstance(main_module._states, TTLCache)
    assert main_module.SESSION_MAX_ENTRIES == 1000
    assert main_module.SESSION_TTL_SECONDS == 86400.0
    assert main_module._states.maxsize == 1000
    assert main_module._states.ttl == 86400.0


def test_ttl_eviction():
    """Verify that sessions expire after their TTL duration."""
    # Test TTL eviction using a small TTL
    short_cache: TTLCache[str, dict] = TTLCache(maxsize=10, ttl=0.1)
    short_cache["session_temp"] = init_state()
    assert "session_temp" in short_cache

    # Wait for TTL to expire
    time.sleep(0.15)
    assert "session_temp" not in short_cache
    assert short_cache.get("session_temp") is None


def test_lru_cap_eviction():
    """Verify that the least recently used session is evicted when capacity is reached."""
    bounded_cache: TTLCache[str, dict] = TTLCache(maxsize=2, ttl=60.0)

    # Insert two sessions
    bounded_cache["sess_1"] = {"id": 1}
    bounded_cache["sess_2"] = {"id": 2}
    assert len(bounded_cache) == 2

    # Insert third session - sess_1 should be evicted as the least recently used
    bounded_cache["sess_3"] = {"id": 3}
    assert "sess_1" not in bounded_cache
    assert "sess_2" in bounded_cache
    assert "sess_3" in bounded_cache

    # Touch sess_2 so sess_3 becomes the least recently used
    _ = bounded_cache["sess_2"]
    bounded_cache["sess_4"] = {"id": 4}

    assert "sess_3" not in bounded_cache
    assert "sess_2" in bounded_cache
    assert "sess_4" in bounded_cache


def test_reset_session_endpoint():
    """Verify that the /sessions/{session_id}/reset endpoint properly removes the session from the cache."""
    client = TestClient(app)
    test_session_id = "test_reset_session_abc"

    # Pre-populate state
    with _lock:
        _states[test_session_id] = init_state()
        assert test_session_id in _states

    response = client.post(f"/sessions/{test_session_id}/reset")
    assert response.status_code == 200
    assert response.json() == {"status": "reset", "session_id": test_session_id}

    with _lock:
        assert test_session_id not in _states


def test_concurrent_access_thread_safety():
    """Verify that concurrent access using _lock maintains cache consistency without errors."""
    test_cache: TTLCache[str, dict] = TTLCache(maxsize=100, ttl=60.0)
    errors = []

    def worker(worker_id: int):
        try:
            for i in range(50):
                sess_id = f"worker_{worker_id}_sess_{i % 5}"
                with _lock:
                    st = test_cache.setdefault(sess_id, init_state())
                    st["turn_count"] = st.get("turn_count", 0) + 1
                    test_cache[sess_id] = st
        except Exception as e:
            errors.append(e)

    threads = [Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
