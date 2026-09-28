from __future__ import annotations

import copy
import logging
from threading import Lock
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.decision import post_update, step
from core.state import init_state
from pipeline import check_llm_health, parse_context
from utils import resolve_provider_and_model_name


logger = logging.getLogger("metamo.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class HistoryTurn(BaseModel):
    query: str | None = None
    answer: str | None = None
    user: str | None = None
    assistant: str | None = None


class RecommendationRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str = Field(default="default", min_length=1)
    recent_history: list[HistoryTurn] = Field(default_factory=list)
    project_context: dict[str, Any] = Field(default_factory=dict)


class ScoreItem(BaseModel):
    action: str
    score: float


class RecommendationResponse(BaseModel):
    request_id: str
    session_id: str
    recommended_action: str
    confidence: float
    reason: str
    signals: dict[str, Any]
    score_top3: list[ScoreItem]
    state: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    provider: str | None = None
    model: str | None = None
    latency_ms: float | None = None
    error: str | None = None


app = FastAPI(
    title="MetaMo Recommendation API",
    description="Action recommendation API for OpenClaw/Qwestor integration.",
    version="0.1.0",
)

_states: dict[str, dict[str, Any]] = {}
_lock = Lock()


def _history_turn_to_parser_turn(turn: HistoryTurn) -> dict[str, str]:
    return {
        "query": str(turn.query if turn.query is not None else turn.user or ""),
        "answer": str(turn.answer if turn.answer is not None else turn.assistant or ""),
    }


def _request_history_to_parser_turns(
    recent_history: list[HistoryTurn],
) -> list[dict[str, str]]:
    turns = [_history_turn_to_parser_turn(turn) for turn in recent_history]
    return [turn for turn in turns if turn["query"].strip() or turn["answer"].strip()]


def _fallback_history_from_state(state: dict[str, Any]) -> list[dict[str, str]]:
    context_history = state.get("context_history", [])
    if not isinstance(context_history, list):
        return []
    turns: list[dict[str, str]] = []
    for item in context_history[-3:]:
        if isinstance(item, dict):
            turns.append(
                {
                    "query": str(item.get("query", "")),
                    "answer": str(item.get("answer", "")),
                }
            )
    return turns


def _signals_from_context(context: dict[str, Any]) -> dict[str, Any]:
    signal_keys = [
        "urgent",
        "complexity",
        "ambiguity",
        "expertise",
        "threshold",
        "topic_familiarity",
        "failure_signal",
        "intent_type",
        "reflective_intent",
        "verify_request",
        "needs_external_evidence",
        "needs_task_plan",
        "needs_multi_source_integration",
        "valence",
        "parser_calibration",
    ]
    return {key: copy.deepcopy(context.get(key)) for key in signal_keys if key in context}


def _score_top3(decision: dict[str, Any]) -> list[ScoreItem]:
    score_items: list[ScoreItem] = []
    for item in decision.get("score_top3", []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            score_items.append(ScoreItem(action=str(item[0]), score=float(item[1])))
    return score_items


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_count": int(state.get("turn_count", 0)),
        "modulators": copy.deepcopy(state.get("modulators", {})),
        "goals": copy.deepcopy(state.get("goals", {})),
        "anti_goals": copy.deepcopy(state.get("anti_goals", {})),
    }


@app.get("/health", response_model=HealthResponse)
def health(
    check_provider: bool = Query(
        default=True,
        description="Whether to verify live connectivity and authentication with the configured LLM provider.",
    ),
) -> JSONResponse:
    if not check_provider:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ok"},
        )

    health_result = check_llm_health()
    if not health_result.get("ok"):
        logger.warning(
            "MetaMo health check failed: provider=%s model=%s error=%s",
            health_result.get("provider"),
            health_result.get("model"),
            health_result.get("error"),
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "provider": health_result.get("provider"),
                "model": health_result.get("model"),
                "latency_ms": health_result.get("latency_ms"),
                "error": health_result.get("error"),
            },
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "provider": health_result.get("provider"),
            "model": health_result.get("model"),
            "latency_ms": health_result.get("latency_ms"),
            "error": None,
        },
    )


@app.post("/recommend-action", response_model=RecommendationResponse)
def recommend_action(request: RecommendationRequest) -> RecommendationResponse:
    load_dotenv(override=True)
    request_id = str(uuid4())
    with _lock:
        state = _states.setdefault(request.session_id, init_state())
        fallback_history = _fallback_history_from_state(state)

    history_turns = _request_history_to_parser_turns(request.recent_history)
    if not history_turns:
        history_turns = fallback_history

    provider_name, model_name = resolve_provider_and_model_name()
    context = parse_context(
        request.query,
        history_turns=history_turns,
        model=model_name,
        provider=provider_name,
    )

    with _lock:
        state = _states.setdefault(request.session_id, init_state())
        decision = step(context, state)
        updated_state = post_update(context, state, decision)
        history = updated_state.get("context_history", [])
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "query": request.query,
                "answer": "",
                "recommended_action": str(decision.get("action", "unknown")),
            }
        )
        updated_state["context_history"] = history[-20:]
        _states[request.session_id] = updated_state
        state_summary = _state_summary(updated_state)

    recommended_action = str(decision.get("action", "unknown"))
    confidence = float(decision.get("confidence", 0.0))
    reason = str(decision.get("reason", ""))

    logger.info(
        "MetaMo recommendation request_id=%s session_id=%s action=%s confidence=%.3f query=%r",
        request_id,
        request.session_id,
        recommended_action,
        confidence,
        request.query,
    )

    return RecommendationResponse(
        request_id=request_id,
        session_id=request.session_id,
        recommended_action=recommended_action,
        confidence=confidence,
        reason=reason,
        signals=_signals_from_context(context),
        score_top3=_score_top3(decision),
        state=state_summary,
    )


@app.post("/sessions/{session_id}/reset")
def reset_session(session_id: str) -> dict[str, str]:
    with _lock:
        _states.pop(session_id, None)
    logger.info("MetaMo session reset session_id=%s", session_id)
    return {"status": "reset", "session_id": session_id}
