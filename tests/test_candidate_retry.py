from benchmark.cases_loader import load_case
from benchmark.runner import _collect_candidate
from benchmark.schema import CandidateAnswer, ModelCallMeta


def _candidate(*, error=None, finish_reason="", answers=None):
    return CandidateAnswer(
        candidate_key="candidate",
        label="Candidate",
        blind_id="Candidate 1",
        answers=answers or {},
        raw_response="response" if answers else "",
        meta=ModelCallMeta(
            model="model",
            provider="openrouter",
            error=error,
            finish_reason=finish_reason,
            cost_usd=0.01,
        ),
    )


def test_candidate_transport_failure_retries_once(monkeypatch):
    case = load_case("caseC")
    responses = [
        _candidate(error="HTTP 503"),
        _candidate(answers={q.id: "answer" for q in case.questions}),
    ]
    calls = []

    def fake_once(*args, **kwargs):
        calls.append(1)
        return responses.pop(0)

    monkeypatch.setattr("benchmark.runner._collect_candidate_once", fake_once)
    result = _collect_candidate(
        case,
        {"key": "candidate", "provider": "openrouter"},
        "Candidate 1",
    )

    assert len(calls) == 2
    assert result.meta.retry_count == 1
    assert result.meta.cost_usd == 0.02


def test_candidate_empty_or_missing_sections_is_not_retried(monkeypatch):
    case = load_case("caseC")
    calls = []

    def fake_once(*args, **kwargs):
        calls.append(1)
        return _candidate(answers={"diagnosis": "only one section"})

    monkeypatch.setattr("benchmark.runner._collect_candidate_once", fake_once)
    result = _collect_candidate(
        case,
        {"key": "candidate", "provider": "openrouter"},
        "Candidate 1",
    )

    assert len(calls) == 1
    assert result.meta.retry_count == 0
