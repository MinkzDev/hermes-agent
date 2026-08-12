"""Tests for Discord ignored_channels and no_thread_channels config."""

from types import SimpleNamespace
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import socket
import sys

import pytest

from gateway.config import Platform, PlatformConfig


def _ensure_discord_mock():
    """Install a mock discord module when discord.py isn't available."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


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


class FakeDMChannel:
    def __init__(self, channel_id: int = 1, name: str = "dm"):
        self.id = channel_id
        self.name = name


class FakeTextChannel:
    def __init__(self, channel_id: int = 1, name: str = "general", guild_name: str = "Hermes Server", guild_id: int = 100):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name=guild_name, id=guild_id)
        self.topic = None


class FakeThread:
    def __init__(self, channel_id: int = 1, name: str = "thread", parent=None, guild_name: str = "Hermes Server"):
        self.id = channel_id
        self.name = name
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", None) or SimpleNamespace(name=guild_name)
        self.topic = None


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    adapter._text_batch_delay_seconds = 0  # disable batching for tests
    adapter.handle_message = AsyncMock()
    return adapter


def make_message(*, channel, content: str, mentions=None):
    author = SimpleNamespace(id=42, display_name="TestUser", name="TestUser", bot=False)
    return SimpleNamespace(
        id=123,
        content=content,
        mentions=list(mentions or []),
        attachments=[],
        message_snapshots=[],
        stickers=[],
        embeds=[],
        reference=None,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        guild=getattr(channel, "guild", None),
        author=author,
        type=discord_platform.discord.MessageType.default,
    )


def bxr_config():
    return PlatformConfig(
        enabled=True,
        token="fixture-token",
        reply_to_mode="all",
        gateway_restart_notification=False,
        typing_indicator=False,
        extra={
            "bxr_operator_ingress": {
                "enabled": True,
                "profile_id": "bxr-operator",
                "allowed_user_id": "42",
                "guild_id": "100",
                "channel_id": "800",
                "authority_granted": False,
            },
            "allow_from": ["42"],
            "allowed_channels": ["800"],
            "allowed_roles": [],
            "free_response_channels": [],
            "allow_all_users": False,
            "auto_thread": False,
            "reactions": False,
            "slash_commands": False,
            "history_backfill": False,
            "missed_message_backfill": False,
            "allow_any_attachment": False,
            "require_mention": True,
            "thread_require_mention": True,
            "allow_bots": "none",
        },
    )


# ── ignored_channels ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ignored_channel_blocks_even_with_mention(adapter, monkeypatch):
    """Ignored channels take priority — even @mentions are dropped."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "500")

    bot_user = adapter._client.user
    message = make_message(
        channel=FakeTextChannel(channel_id=500),
        content=f"<@{bot_user.id}> hello",
        mentions=[bot_user],
    )
    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_ignored_channel_processes_normally(adapter, monkeypatch):
    """Channels not in the ignored list process normally."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "500,600")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    # Stub auto-thread creation so this test focuses on ignored-channel
    # routing only — auto-thread failures now correctly skip agent invocation
    # (#20243), which would otherwise mask the assertion below.
    adapter._auto_create_thread = AsyncMock(return_value=FakeThread(channel_id=999))

    message = make_message(channel=FakeTextChannel(channel_id=700), content="hello")
    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_ignored_channels_empty_string_ignores_nothing(adapter, monkeypatch):
    """Empty DISCORD_IGNORED_CHANNELS means nothing is ignored."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    # Stub auto-thread creation so this test focuses on ignored-channel
    # routing only — auto-thread failures now correctly skip agent invocation
    # (#20243), which would otherwise mask the assertion below.
    adapter._auto_create_thread = AsyncMock(return_value=FakeThread(channel_id=999))

    message = make_message(channel=FakeTextChannel(channel_id=500), content="hello")
    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


# ── no_thread_channels ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_thread_channel_skips_auto_thread(adapter, monkeypatch):
    """Channels in no_thread_channels should not auto-create threads."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_NO_THREAD_CHANNELS", "800")
    monkeypatch.delenv("DISCORD_AUTO_THREAD", raising=False)
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    adapter._auto_create_thread = AsyncMock(return_value=FakeThread(channel_id=999))

    message = make_message(channel=FakeTextChannel(channel_id=800), content="hello")
    await adapter._handle_message(message)

    adapter._auto_create_thread.assert_not_awaited()
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_type == "group"


# ── auto-thread failure must not silently fall back to inline (#20243) ──


@pytest.mark.asyncio
async def test_auto_thread_failure_skips_agent_and_notifies_user(adapter, monkeypatch):
    """Auto-thread creation failure must not trigger an inline parent-channel reply.

    Before #20243, ``effective_channel = auto_threaded_channel or message.channel``
    silently routed the response back to the parent channel when thread creation
    failed, breaking thread-first Discord workflows. The fix surfaces a short
    visible error to the parent channel and skips agent invocation entirely so
    the user can retry.
    """
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")
    monkeypatch.delenv("DISCORD_NO_THREAD_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    adapter._auto_create_thread = AsyncMock(return_value=None)

    channel = FakeTextChannel(channel_id=800)
    channel.send = AsyncMock()
    message = make_message(channel=channel, content="hello")
    await adapter._handle_message(message)

    adapter._auto_create_thread.assert_awaited_once()
    # Agent must NOT be invoked when the routing target failed.
    adapter.handle_message.assert_not_awaited()
    # User gets a visible explanation in the parent channel instead of a silent
    # inline reply.
    channel.send.assert_awaited_once()
    sent_text = channel.send.await_args.args[0]
    assert "could not create" in sent_text.lower()
    assert "thread" in sent_text.lower()


# ── config.py bridging ───────────────────────────────────────────────


def test_config_bridges_ignored_channels(monkeypatch, tmp_path):
    """gateway/config.py bridges discord.ignored_channels to env var."""
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "discord": {
            "ignored_channels": ["111", "222"],
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Use setenv (not delenv) so monkeypatch registers cleanup even when
    # the var doesn't exist yet — load_gateway_config will overwrite it.
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "")

    from gateway.config import load_gateway_config
    load_gateway_config()

    import os
    assert os.getenv("DISCORD_IGNORED_CHANNELS") == "111,222"


def test_config_preserves_allow_all_users_boolean_on_real_yaml_load(monkeypatch, tmp_path):
    """The real YAML loader keeps the closed gate typed while bridging legacy env."""
    import os
    import yaml

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "gateway": {
                    "platforms": {
                        "discord": {
                            "enabled": True,
                            "extra": {"allow_all_users": False},
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "")

    from gateway.config import load_gateway_config

    config = load_gateway_config()

    assert config.platforms[Platform.DISCORD].extra["allow_all_users"] is False
    assert (os.getenv("DISCORD_ALLOW_ALL_USERS") or "").lower() not in {
        "true",
        "1",
        "yes",
    }


@pytest.fixture
def bxr_adapter(monkeypatch, tmp_path):
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    instance = DiscordAdapter(bxr_config())
    bot = SimpleNamespace(id=999, bot=True)
    instance._client = SimpleNamespace(user=bot)
    return instance


def test_bxr_policy_is_closed_and_rejects_wildcards_and_roles(bxr_adapter):
    assert bxr_adapter._bxr_operator_policy_error() is None

    bxr_adapter.config.extra["allowed_channels"] = ["*"]
    assert bxr_adapter._bxr_operator_policy_error() == "bxr_allowed_channels"
    bxr_adapter.config.extra["allowed_channels"] = ["800"]

    bxr_adapter.config.extra["allowed_roles"] = ["55"]
    assert bxr_adapter._bxr_operator_policy_error() == "bxr_roles"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda message, bot: setattr(message.author, "id", 43), "USER_NOT_ALLOWLISTED"),
        (lambda message, bot: setattr(message.guild, "id", 101), "GUILD_NOT_ALLOWLISTED"),
        (lambda message, bot: setattr(message.channel, "id", 801), "CHANNEL_NOT_ALLOWLISTED"),
        (
            lambda message, bot: (
                setattr(message, "mentions", []),
                setattr(message, "content", "BXR status"),
            ),
            "EXPLICIT_MENTION_REQUIRED",
        ),
        (lambda message, bot: message.attachments.append(SimpleNamespace()), "ATTACHMENT_PROHIBITED"),
        (lambda message, bot: message.message_snapshots.append(SimpleNamespace()), "SNAPSHOT_PROHIBITED"),
        (lambda message, bot: setattr(message, "content", f"<@{bot.id}> /status"), "SLASH_COMMAND_PROHIBITED"),
        (lambda message, bot: setattr(message, "content", f"<@{bot.id}> " + "x" * 1001), "INPUT_TOO_LONG"),
    ],
)
def test_bxr_ingress_denies_every_nonmatching_origin_or_shape(bxr_adapter, mutation, reason):
    bot = bxr_adapter._client.user
    message = make_message(
        channel=FakeTextChannel(channel_id=800),
        content=f"<@{bot.id}> BXR status",
        mentions=[bot],
    )
    mutation(message, bot)
    admitted, observed_reason, _ = bxr_adapter._bxr_operator_preflight(message)
    assert admitted is False
    assert observed_reason == reason


def test_bxr_admission_records_self_and_nontext_denials(bxr_adapter):
    bot = bxr_adapter._client.user
    message = make_message(
        channel=FakeTextChannel(channel_id=800),
        content=f"<@{bot.id}> BXR status",
        mentions=[bot],
    )
    bxr_adapter._write_bxr_origin_event = MagicMock(return_value="event-id")

    message.author = bot
    assert bxr_adapter._discord_message_admission(message, claim=False) == (False, False)
    assert bxr_adapter._write_bxr_origin_event.call_args.kwargs["reason"] == "BOT_SELF"

    message.author = SimpleNamespace(id=42, bot=False)
    message.type = object()
    assert bxr_adapter._discord_message_admission(message, claim=False) == (False, False)
    assert (
        bxr_adapter._write_bxr_origin_event.call_args.kwargs["reason"]
        == "MESSAGE_TYPE_PROHIBITED"
    )


def test_bxr_thread_requires_mention_and_exact_parent_channel(bxr_adapter):
    bot = bxr_adapter._client.user
    parent = FakeTextChannel(channel_id=800)
    thread = FakeThread(channel_id=900, parent=parent)
    message = make_message(channel=thread, content=f"<@{bot.id}> Continue BXR", mentions=[bot])

    admitted, reason, normalized = bxr_adapter._bxr_operator_preflight(message)
    assert (admitted, reason, normalized) == (True, "ADMITTED", "Continue BXR")

    message.mentions = []
    message.content = "Continue BXR"
    assert bxr_adapter._bxr_operator_preflight(message)[1] == "EXPLICIT_MENTION_REQUIRED"


def _bxr_runner(*, active_profile="bxr-operator", multiplex_profiles=False):
    return SimpleNamespace(
        config=SimpleNamespace(multiplex_profiles=multiplex_profiles),
        _active_profile_name=MagicMock(return_value=active_profile),
    )


def _matching_bxr_message(adapter):
    bot = adapter._client.user
    return make_message(
        channel=FakeTextChannel(channel_id=800),
        content=f"<@{bot.id}> BXR status",
        mentions=[bot],
    )


@pytest.mark.asyncio
async def test_bxr_single_profile_runner_stamps_and_admits_source(bxr_adapter):
    bxr_adapter.gateway_runner = _bxr_runner()
    bxr_adapter.handle_message = AsyncMock()
    bxr_adapter._write_bxr_origin_event = MagicMock(return_value="event-id")

    assert await bxr_adapter._handle_bxr_operator_message(
        _matching_bxr_message(bxr_adapter)
    ) is True

    event = bxr_adapter.handle_message.await_args.args[0]
    assert event.source.profile == "bxr-operator"
    assert event.metadata == {
        "bxr_operator_ingress": True,
        "origin_event_id": "event-id",
        "authority_granted": False,
    }


@pytest.mark.parametrize(
    "runner",
    [
        None,
        _bxr_runner(active_profile="default"),
        _bxr_runner(multiplex_profiles=True),
    ],
    ids=["absent-runner", "wrong-profile", "multiplex-mismatch"],
)
@pytest.mark.asyncio
async def test_bxr_profile_ownership_mismatch_denies_without_dispatch(bxr_adapter, runner):
    bxr_adapter.gateway_runner = runner
    bxr_adapter.handle_message = AsyncMock()
    bxr_adapter._write_bxr_origin_event = MagicMock(return_value="event-id")

    assert await bxr_adapter._handle_bxr_operator_message(
        _matching_bxr_message(bxr_adapter)
    ) is False

    bxr_adapter.handle_message.assert_not_awaited()
    assert bxr_adapter._write_bxr_origin_event.call_args.kwargs["status"] == "DENIED"
    assert (
        bxr_adapter._write_bxr_origin_event.call_args.kwargs["reason"]
        == "PROFILE_ROUTE_MISMATCH"
    )


@pytest.mark.asyncio
async def test_bxr_profile_mismatch_persists_metadata_only_denial(bxr_adapter):
    bxr_adapter.gateway_runner = _bxr_runner(active_profile="default")
    bxr_adapter.handle_message = AsyncMock()
    message = _matching_bxr_message(bxr_adapter)

    assert await bxr_adapter._handle_bxr_operator_message(message) is False

    stored_files = list(bxr_adapter._bxr_origin_event_root().glob("*.json"))
    assert len(stored_files) == 1
    stored = stored_files[0].read_text(encoding="utf-8")
    assert "BXR status" not in stored
    assert "content" not in stored
    assert "PROFILE_ROUTE_MISMATCH" in stored
    assert '"authority_granted":false' in stored
    bxr_adapter.handle_message.assert_not_awaited()


def test_bxr_origin_event_is_content_free_immutable_and_bounded(bxr_adapter, monkeypatch, tmp_path):
    bot = bxr_adapter._client.user
    secret_text = "request text that must never persist"
    message = make_message(
        channel=FakeTextChannel(channel_id=800),
        content=f"<@{bot.id}> {secret_text}",
        mentions=[bot],
    )
    event_id = bxr_adapter._write_bxr_origin_event(
        message,
        status="ADMITTED",
        reason="POLICY_MATCH",
        normalized_length=len(secret_text),
        reserve_after=1,
    )
    assert event_id is not None
    root = bxr_adapter._bxr_origin_event_root()
    stored = (root / f"{event_id}.json").read_text(encoding="utf-8")
    assert secret_text not in stored
    assert "content" not in stored
    assert "hash" not in stored
    assert bxr_adapter._write_bxr_origin_event(
        message,
        status="ADMITTED",
        reason="POLICY_MATCH",
        normalized_length=len(secret_text),
        reserve_after=1,
    ) == event_id

    monkeypatch.setattr(discord_platform, "_BXR_DISCORD_EVENT_LIMIT", 1)
    message.id = 124
    assert bxr_adapter._write_bxr_origin_event(
        message,
        status="DENIED",
        reason="TEST_BOUND",
    ) is None


@pytest.mark.asyncio
async def test_bxr_reply_is_bounded_and_every_chunk_is_anchored(bxr_adapter, monkeypatch):
    channel = FakeTextChannel(channel_id=800)
    channel.send = AsyncMock(side_effect=[SimpleNamespace(id=index) for index in range(1, 5)])
    source = SimpleNamespace(chat_id="800", guild_id="100")
    event = SimpleNamespace(
        raw_message=SimpleNamespace(channel=channel),
        source=source,
        message_id="123",
    )
    reference = object()
    monkeypatch.setattr(discord_platform.discord, "MessageReference", MagicMock(return_value=reference))

    result = await bxr_adapter._send_bxr_operator_reply(event, "x" * 8000)
    assert result.success is True
    assert channel.send.await_count == 4
    assert all(call.kwargs["reference"] is reference for call in channel.send.await_args_list)
    assert all(len(call.kwargs["content"]) <= 2000 for call in channel.send.await_args_list)

    channel.send.reset_mock()
    assert not (await bxr_adapter._send_bxr_operator_reply(event, "x" * 8001)).success
    channel.send.assert_not_awaited()

    # Discord's platform limit is UTF-16 units, not Python code points.
    channel.send.side_effect = [SimpleNamespace(id=index) for index in range(5, 9)]
    emoji_result = await bxr_adapter._send_bxr_operator_reply(event, "\U0001f600" * 4000)
    assert emoji_result.success
    assert channel.send.await_count == 4
    for call in channel.send.await_args_list:
        assert discord_platform.utf16_len(call.kwargs["content"]) == 2000


@pytest.mark.asyncio
async def test_bxr_reply_anchor_failure_has_no_unanchored_fallback(bxr_adapter, monkeypatch):
    channel = FakeTextChannel(channel_id=800)
    channel.send = AsyncMock(side_effect=RuntimeError("unknown message"))
    event = SimpleNamespace(
        raw_message=SimpleNamespace(channel=channel),
        source=SimpleNamespace(chat_id="800", guild_id="100"),
        message_id="123",
    )
    monkeypatch.setattr(discord_platform.discord, "MessageReference", MagicMock(return_value=object()))

    result = await bxr_adapter._send_bxr_operator_reply(event, "BXR status")
    assert result.success is False
    assert channel.send.await_count == 1
    assert channel.send.await_args.kwargs["reference"] is not None


@pytest.mark.asyncio
async def test_bxr_generic_send_rail_is_closed(bxr_adapter):
    result = await bxr_adapter.send("800", "proactive message")
    assert result.success is False
    assert "origin-anchored" in result.error
