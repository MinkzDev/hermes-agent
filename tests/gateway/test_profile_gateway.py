"""Closed profile-scoped gateway invariants for the BXR Discord lane."""

import datetime as dt
import json
import logging
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.run import (
    TurnRunner,
    _BXR_OPERATOR_DISCORD_TOOLS,
    _BXR_OPERATOR_DISCORD_TOOLSETS,
    _BXR_METADATA_ONLY_LOG_MARKER,
    _bxr_expected_tool_for_message,
    _build_gateway_agent_history,
    _bxr_invoked_tool_names,
    _bxr_operator_session_metadata,
    _bxr_operator_session_transcript_row,
    _bxr_operator_tool_surface_is_exact,
    _bxr_projected_tool_definitions,
    _run_bxr_operator_projected_conversation,
    _gateway_tool_definition_names,
    _is_bxr_operator_discord_source,
    _require_bxr_operator_discord_toolsets,
)
from gateway.config import Platform
from gateway.turn_context import TurnContext


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    attempts = []
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def is_loopback(address):
        return isinstance(address, tuple) and str(address[0]) in {"127.0.0.1", "::1"}

    def denied(*args, **kwargs):
        attempts.append((args, kwargs))
        raise AssertionError("offline BXR Hermes test attempted network access")

    def guarded_create_connection(address, *args, **kwargs):
        if is_loopback(address):
            return original_create_connection(address, *args, **kwargs)
        return denied(address, *args, **kwargs)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if str(host) in {"127.0.0.1", "::1", "localhost"}:
            return original_getaddrinfo(host, *args, **kwargs)
        return denied(host, *args, **kwargs)

    def guarded_connect(sock, address):
        if is_loopback(address):
            return original_connect(sock, address)
        return denied(sock, address)

    def guarded_connect_ex(sock, address):
        if is_loopback(address):
            return original_connect_ex(sock, address)
        return denied(sock, address)

    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    yield
    assert attempts == []


def bxr_source(**overrides):
    values = {
        "platform": SimpleNamespace(value="discord"),
        "profile": "bxr-operator",
        "_bxr_operator_ingress": True,
        "message_id": "123",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_bxr_lane_requires_transport_profile_and_explicit_marker():
    assert _is_bxr_operator_discord_source(bxr_source()) is True
    assert _is_bxr_operator_discord_source(bxr_source(profile="default")) is False
    assert _is_bxr_operator_discord_source(
        bxr_source(_bxr_operator_ingress=False)
    ) is False
    assert _is_bxr_operator_discord_source(
        bxr_source(platform=SimpleNamespace(value="telegram"))
    ) is False


def test_closed_tool_schema_extraction_accepts_only_three_existing_bxr_tools():
    definitions = [
        {"name": "mcp__bxr_operator__bxr_status"},
        {"function": {"name": "mcp__bxr_operator__bxr_route"}},
        {"name": "mcp__bxr_operator__bxr_continue_plan"},
    ]
    names = _gateway_tool_definition_names(definitions)
    assert names == _BXR_OPERATOR_DISCORD_TOOLS
    assert _bxr_operator_tool_surface_is_exact(definitions) is True
    assert _bxr_operator_tool_surface_is_exact(definitions[:2]) is False
    assert _bxr_operator_tool_surface_is_exact(
        definitions + [{"name": "terminal"}]
    ) is False
    assert "terminal" not in names
    assert "send_message" not in names


def test_closed_lane_preserves_only_the_exact_bxr_operator_toolset():
    resolved = ["bxr_operator"]

    assert _require_bxr_operator_discord_toolsets(resolved) is resolved
    assert resolved == _BXR_OPERATOR_DISCORD_TOOLSETS


@pytest.mark.parametrize(
    "resolved",
    [
        [],
        ["clarify"],
        ["bxr_operator", "clarify"],
    ],
)
def test_closed_lane_rejects_missing_or_extra_toolsets(resolved):
    with pytest.raises(RuntimeError, match="Closed BXR Discord toolset mismatch"):
        _require_bxr_operator_discord_toolsets(resolved)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("BXR status", "mcp__bxr_operator__bxr_status"),
        ("  bxr   STATUS?!  ", "mcp__bxr_operator__bxr_status"),
        ("What needs my attention?", "mcp__bxr_operator__bxr_status"),
        ("Continue BXR", "mcp__bxr_operator__bxr_continue_plan"),
        ("summarize this repository", None),
    ],
)
def test_closed_lane_maps_only_finite_acceptance_requests(message, expected):
    assert _bxr_expected_tool_for_message(message) == expected


def _bxr_tool_definitions():
    return [
        {"name": "mcp__bxr_operator__bxr_status"},
        {"function": {"name": "mcp__bxr_operator__bxr_route"}},
        {"name": "mcp__bxr_operator__bxr_continue_plan"},
    ]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("BXR status", "mcp__bxr_operator__bxr_status"),
        ("What needs my attention?", "mcp__bxr_operator__bxr_status"),
        ("Continue BXR", "mcp__bxr_operator__bxr_continue_plan"),
    ],
)
def test_closed_lane_projects_one_tool_and_restores_full_surface(message, expected):
    original = _bxr_tool_definitions()
    agent = SimpleNamespace(tools=original)
    observed = {}

    def conversation_call():
        observed["tools"] = list(agent.tools)
        return {
            "messages": [
                {"role": "assistant", "tool_calls": [{"function": {"name": expected}}]}
            ]
        }

    result = _run_bxr_operator_projected_conversation(
        agent,
        _bxr_expected_tool_for_message(message),
        conversation_call,
    )

    assert result["messages"]
    assert _gateway_tool_definition_names(observed["tools"]) == {expected}
    assert agent.tools is original
    assert _gateway_tool_definition_names(agent.tools) == _BXR_OPERATOR_DISCORD_TOOLS


def test_closed_lane_restores_full_surface_after_exception():
    original = _bxr_tool_definitions()
    agent = SimpleNamespace(tools=original)

    def conversation_call():
        assert _gateway_tool_definition_names(agent.tools) == {
            "mcp__bxr_operator__bxr_status"
        }
        raise RuntimeError("synthetic provider failure")

    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        _run_bxr_operator_projected_conversation(
            agent,
            "mcp__bxr_operator__bxr_status",
            conversation_call,
        )
    assert agent.tools is original
    assert _gateway_tool_definition_names(agent.tools) == _BXR_OPERATOR_DISCORD_TOOLS


@pytest.mark.parametrize(
    "invoked",
    [
        [],
        ["mcp__bxr_operator__bxr_route"],
        [
            "mcp__bxr_operator__bxr_route",
            "mcp__bxr_operator__bxr_status",
        ],
    ],
)
def test_closed_status_lane_fails_before_return_on_wrong_tool_set(invoked):
    agent = SimpleNamespace(tools=_bxr_tool_definitions())
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": name}} for name in invoked],
        }
    ]

    with pytest.raises(RuntimeError, match="invoked tool mismatch"):
        _run_bxr_operator_projected_conversation(
            agent,
            "mcp__bxr_operator__bxr_status",
            lambda: {"messages": messages},
        )


def test_closed_lane_redacts_turn_log_without_changing_provider_input(caplog):
    sentinel = "raw-content-sentinel-must-not-persist"
    original = _bxr_tool_definitions()
    agent = SimpleNamespace(tools=original)
    logger = logging.getLogger("agent.turn_context")
    observed = {}

    def conversation_call():
        observed["provider_input"] = sentinel
        logger.info(
            "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
            "session",
            "model",
            "provider",
            "discord",
            0,
            sentinel,
        )
        return {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "mcp__bxr_operator__bxr_status"}}
                    ],
                }
            ]
        }

    with caplog.at_level(logging.INFO, logger="agent.turn_context"):
        _run_bxr_operator_projected_conversation(
            agent,
            "mcp__bxr_operator__bxr_status",
            conversation_call,
        )

    assert observed["provider_input"] == sentinel
    assert sentinel not in caplog.text
    assert _BXR_METADATA_ONLY_LOG_MARKER in caplog.text


def test_non_bxr_turn_log_is_unchanged(caplog):
    sentinel = "non-bxr-log-sentinel"
    logger = logging.getLogger("agent.turn_context")
    with caplog.at_level(logging.INFO, logger="agent.turn_context"):
        logger.info(
            "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
            "session",
            "model",
            "provider",
            "cli",
            0,
            sentinel,
        )
    assert sentinel in caplog.text


def test_invoked_tool_projection_keeps_names_without_arguments_or_content():
    messages = [
        {
            "role": "assistant",
            "content": "raw assistant content",
            "tool_calls": [
                {
                    "function": {
                        "name": "mcp__bxr_operator__bxr_status",
                        "arguments": '{"raw":"must not persist"}',
                    }
                },
                {"function": {"name": "terminal", "arguments": "whoami"}},
            ],
        }
    ]
    assert _bxr_invoked_tool_names(messages) == [
        "mcp__bxr_operator__bxr_status"
    ]


def test_bxr_progress_callback_records_only_tool_identity_and_emits_nothing():
    adapter = SimpleNamespace(_bxr_note_tool=MagicMock(return_value=True))
    runner = SimpleNamespace(_adapter_for_source=MagicMock(return_value=adapter))
    context = SimpleNamespace(source=bxr_source())
    turn = TurnRunner(runner, context)

    turn.progress_callback(
        "tool.started",
        tool_name="mcp__bxr_operator__bxr_status",
        preview="raw request preview",
        args={"raw": "request"},
    )

    adapter._bxr_note_tool.assert_called_once_with(
        context.source,
        "mcp__bxr_operator__bxr_status",
    )


def test_bxr_session_metadata_is_closed_content_free_and_authority_negative():
    source = bxr_source(
        user_id="700",
        guild_id="100",
        chat_id="800",
        parent_chat_id="",
    )
    event = SimpleNamespace(
        message_id="900",
        timestamp=dt.datetime(2026, 8, 11, tzinfo=dt.timezone.utc),
        text="raw request must not persist",
    )
    payload = _bxr_operator_session_metadata(
        source,
        event,
        ["mcp__bxr_operator__bxr_status", "terminal"],
    )

    assert set(payload) == {
        "role",
        "event",
        "profile",
        "actor",
        "origin",
        "received_at",
        "recorded_at",
        "admission",
        "invoked_bxr_tools",
        "authority_granted",
    }
    assert payload["admission"] == "ADMITTED"
    assert payload["invoked_bxr_tools"] == ["mcp__bxr_operator__bxr_status"]
    assert payload["authority_granted"] is False
    encoded = json.dumps(payload, sort_keys=True)
    assert "raw request must not persist" not in encoded
    assert "raw response must not persist" not in encoded
    assert "terminal" not in encoded


def test_bxr_session_metadata_round_trips_in_real_db_and_is_replay_inert(tmp_path):
    from hermes_state import SessionDB

    source = bxr_source(
        user_id="700",
        guild_id="100",
        chat_id="800",
        parent_chat_id="",
    )
    event = SimpleNamespace(
        message_id="900",
        timestamp=dt.datetime(2026, 8, 11, tzinfo=dt.timezone.utc),
        text="raw request must not persist",
    )
    row = _bxr_operator_session_transcript_row(
        source,
        event,
        ["mcp__bxr_operator__bxr_status", "terminal"],
    )
    expected = _bxr_operator_session_metadata(
        source,
        event,
        ["mcp__bxr_operator__bxr_status", "terminal"],
    )
    assert row == {"role": "session_meta", "content": expected}

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        session_id = "bxr-session-metadata-round-trip"
        db.create_session(session_id, source="discord")
        db.append_message(
            session_id=session_id,
            role=row["role"],
            content=row["content"],
        )
        stored = db.get_messages(session_id)
    finally:
        db.close()

    assert len(stored) == 1
    assert stored[0]["role"] == "session_meta"
    assert stored[0]["content"] == expected
    assert stored[0]["content"]["invoked_bxr_tools"] == [
        "mcp__bxr_operator__bxr_status"
    ]
    assert stored[0]["content"]["authority_granted"] is False
    assert _build_gateway_agent_history(stored) == ([], None)
    encoded = json.dumps(stored, sort_keys=True)
    assert "raw request must not persist" not in encoded
    assert "raw response must not persist" not in encoded
    assert "terminal" not in encoded


class _StopClosedLaneRun(RuntimeError):
    pass


def _exercise_turn_runner_until_conversation(
    *, bxr_closed_lane, constructor_probe=None
):
    session_db = SimpleNamespace(marker="session-db")
    runner = MagicMock()
    runner._get_system_prompt_for_channel.return_value = None
    runner._resolve_session_agent_runtime.return_value = (
        "fixture-model",
        {"provider": "fixture-provider"},
    )
    runner._provider_routing = {}
    runner._resolve_session_reasoning_config.return_value = None
    runner._resolve_session_service_tier.return_value = None
    runner._service_tier = None
    runner.config = SimpleNamespace(
        streaming=SimpleNamespace(enabled=False, transport="off")
    )
    runner._resolve_turn_agent_config.return_value = {
        "model": "fixture-model",
        "runtime": {"provider": "fixture-provider"},
        "request_overrides": {},
    }
    runner._agent_config_signature.return_value = ("fixture-signature",)
    runner._extract_cache_busting_config.return_value = {}
    runner._agent_cache_lock = None
    runner._agent_cache = None
    runner._session_db = session_db
    runner._prefill_messages = None
    runner._refresh_fallback_model.return_value = None
    runner._consume_pending_turn_sidecar_notes.return_value = []
    runner._adapter_for_source.return_value = None
    runner._pending_model_notes = {}
    runner._consume_pending_native_image_paths.return_value = []
    runner._attach_session_title_callback.return_value = None
    runner.session_store = SimpleNamespace(_entries={})

    source = SimpleNamespace(
        platform=Platform.DISCORD,
        profile="bxr-operator" if bxr_closed_lane else "default",
        _bxr_operator_ingress=bxr_closed_lane,
        user_id="fixture-user",
        user_id_alt=None,
        user_name="fixture-operator",
        chat_id="fixture-channel",
        chat_name="fixture-chat",
        chat_type="channel",
        thread_id=None,
        parent_chat_id=None,
    )
    observed = {}

    class FixtureAgent:
        def __init__(self, **kwargs):
            if constructor_probe is not None:
                constructor_probe()
            observed["constructor"] = kwargs
            self.tools = [
                {"name": name}
                for name in sorted(_BXR_OPERATOR_DISCORD_TOOLS)
            ] if bxr_closed_lane else []
            self.session_id = None

        def run_conversation(self, message, **kwargs):
            observed["message"] = message
            observed["conversation"] = kwargs
            observed["agent"] = self
            raise _StopClosedLaneRun

    context = TurnContext(
        source=source,
        _run_still_current=lambda: True,
        message="fixture request",
        history=[],
        session_id=None,
        session_key="fixture-session",
        user_config={
            "gateway": {
                "platforms": {"discord": {"skip_context_files": False}}
            },
            "display": {},
        },
        enabled_toolsets=list(_BXR_OPERATOR_DISCORD_TOOLSETS),
        disabled_toolsets=[],
        AIAgent=FixtureAgent,
        resolve_display_setting=lambda *_args, **_kwargs: None,
        _hooks_ref=SimpleNamespace(loaded_hooks=False),
    )

    with pytest.raises(_StopClosedLaneRun):
        TurnRunner(runner, context).run_sync()
    return observed, session_db


def test_turn_runner_closed_lane_suppresses_models_dev_before_construction(
    monkeypatch,
):
    from agent import models_dev

    events = []
    original_setter = models_dev.set_automatic_refresh_enabled

    def record_refresh_state(enabled):
        events.append(("automatic_refresh", enabled))
        original_setter(enabled)

    def reject_network_path(*_args, **_kwargs):
        raise AssertionError("closed BXR lane started a models.dev network path")

    def probe_model_metadata():
        events.append(
            (
                "agent_construction",
                models_dev._models_dev_automatic_refresh_enabled,
            )
        )
        assert models_dev.fetch_models_dev() == {}

    monkeypatch.setattr(models_dev, "_models_dev_automatic_refresh_enabled", True)
    monkeypatch.setattr(models_dev, "_models_dev_cache", {})
    monkeypatch.setattr(models_dev, "_load_disk_cache", lambda: {})
    monkeypatch.setattr(
        models_dev,
        "_start_background_refresh_models_dev",
        reject_network_path,
    )
    monkeypatch.setattr(
        models_dev,
        "_fetch_models_dev_from_network",
        reject_network_path,
    )
    monkeypatch.setattr(models_dev, "set_automatic_refresh_enabled", record_refresh_state)

    observed, _session_db = _exercise_turn_runner_until_conversation(
        bxr_closed_lane=True,
        constructor_probe=probe_model_metadata,
    )

    assert events == [
        ("automatic_refresh", False),
        ("agent_construction", False),
    ]
    assert _bxr_operator_tool_surface_is_exact(observed["agent"].tools) is True


def test_turn_runner_closed_lane_applies_all_non_persistence_guards():
    observed, _session_db = _exercise_turn_runner_until_conversation(
        bxr_closed_lane=True
    )
    constructor = observed["constructor"]
    agent = observed["agent"]

    assert constructor["session_db"] is None
    assert constructor["skip_context_files"] is True
    assert constructor["skip_memory"] is True
    assert constructor["skip_background_review"] is True
    assert _bxr_operator_tool_surface_is_exact(agent.tools) is True
    assert agent.tool_progress_callback is None
    assert agent.tool_start_callback is None
    assert agent.step_callback is None
    assert agent.stream_delta_callback is None
    assert agent.interim_assistant_callback is None
    assert agent.status_callback is None
    assert agent.event_callback is None
    assert agent.background_review_callback is None
    assert agent.clarify_callback is None
    assert agent.memory_notifications == "off"
    assert agent.thinking_progress is False
    assert _bxr_operator_session_metadata(
        SimpleNamespace(
            profile="bxr-operator",
            user_id="fixture-user",
            guild_id="fixture-guild",
            chat_id="fixture-channel",
            parent_chat_id="",
        ),
        SimpleNamespace(message_id="fixture-message", timestamp=None),
        [],
    )["authority_granted"] is False


def test_turn_runner_non_bxr_lane_keeps_normal_persistence_configuration(
    monkeypatch,
):
    from agent import models_dev

    refresh_calls = []
    monkeypatch.setattr(
        models_dev,
        "set_automatic_refresh_enabled",
        lambda enabled: refresh_calls.append(enabled),
    )
    observed, session_db = _exercise_turn_runner_until_conversation(
        bxr_closed_lane=False
    )
    constructor = observed["constructor"]

    assert refresh_calls == []
    assert constructor["session_db"] is session_db
    assert constructor["skip_context_files"] is False
    assert constructor["skip_memory"] is False
    assert constructor["skip_background_review"] is False
