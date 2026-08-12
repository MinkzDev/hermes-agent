"""Closed profile-scoped gateway invariants for the BXR Discord lane."""

import datetime as dt
import json
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.run import (
    TurnRunner,
    _BXR_OPERATOR_DISCORD_TOOLS,
    _bxr_invoked_tool_names,
    _bxr_operator_session_metadata,
    _bxr_operator_tool_surface_is_exact,
    _gateway_tool_definition_names,
    _is_bxr_operator_discord_source,
)


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
