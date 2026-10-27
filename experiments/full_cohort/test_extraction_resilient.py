"""Exercise the operational repair used after observed Windows sharing failures."""
from unittest.mock import patch

import pytest
from extraction_resilient import with_sharing_retry


def denied(code):
    error = PermissionError("shared file")
    error.winerror = code
    return error


def test_retries_transient_sharing_failure_without_changing_result():
    calls = []

    def write(path, payload):
        calls.append((path, payload))
        if len(calls) < 3:
            raise denied(32)
        return "persisted"

    with patch("extraction_resilient.time.sleep") as sleep:
        assert with_sharing_retry(write, "progress.json", {"completed": 8}) == "persisted"
    assert calls == [("progress.json", {"completed": 8})] * 3
    assert sleep.call_count == 2


def test_does_not_retry_unrelated_permission_error():
    with patch("extraction_resilient.time.sleep") as sleep:
        with pytest.raises(PermissionError):
            with_sharing_retry(lambda: (_ for _ in ()).throw(denied(13)))
    sleep.assert_not_called()


def test_persistent_failure_has_bounded_attempts():
    attempts = []

    def write():
        attempts.append(1)
        raise denied(5)

    with patch("extraction_resilient.time.sleep") as sleep:
        with pytest.raises(PermissionError):
            with_sharing_retry(write)
    assert len(attempts) == 30
    assert sleep.call_count == 29
