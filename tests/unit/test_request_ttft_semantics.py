import pytest

from frontier.entities.request import Request


def test_ttft_ends_at_first_token_and_prefill_latency_remains_available() -> None:
    request = Request(arrived_at=1.0, num_prefill_tokens=8, num_decode_tokens=3)
    request._prefill_completed_at = 1.4

    assert request.prefill_latency == pytest.approx(0.4)
    assert request.ttft == 0

    request.mark_first_decode_token_complete(1.7)

    assert request.ttft == pytest.approx(0.7)
    assert request.prefill_latency == pytest.approx(0.4)
