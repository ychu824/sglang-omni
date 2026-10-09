# SPDX-License-Identifier: Apache-2.0
"""Session runtime contracts that depend on scheduling the wire cannot pin."""

from __future__ import annotations

import asyncio
import base64
import json
import logging

import pytest

from sglang_omni.serve.realtime.control import Closed, Drained, Failure, UnitCompleted
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    ContextLimitError,
    OutputEvent,
    ResponseFinished,
    ResponseStarted,
)
from sglang_omni.serve.realtime.output_buffer import OutputBuffer
from sglang_omni.serve.realtime.protocol import SharedRealtimeSession
from sglang_omni.serve.realtime.runtime import SessionRuntime
from sglang_omni.serve.realtime.schema import JsonObject, SessionConfiguration
from sglang_omni.serve.realtime.types import (
    Capabilities,
    Envelope,
    InteractionAdapter,
    OutputBudgetError,
    OutputSink,
    RuntimeLimits,
    Unit,
)

MODEL_NAME = "duplex-test"
SAMPLE_RATE = 16000
NATIVE_UNIT_MS = 20
UNIT_BYTES = SAMPLE_RATE * NATIVE_UNIT_MS // 1000 * 2
RUNTIME_LOGGER_NAME = "sglang_omni.serve.realtime.runtime"
MAX_STALLED_FRAMES = 16


class GatedAdapter(InteractionAdapter):
    """Blocks on the first unit until released, emitting a fixed reply per unit."""

    def __init__(self, reply: list[OutputEvent]) -> None:
        self.reply = reply
        self.units: list[Unit] = []
        self.output_sink: OutputSink | None = None
        self.has_started = asyncio.Event()
        self.release = asyncio.Event()

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        self.output_sink = emit

    async def process(self, unit: Unit) -> int:
        assert self.output_sink is not None
        self.units.append(unit)
        for event in self.reply:
            await self.output_sink(event)
        self.has_started.set()
        await self.release.wait()
        return unit.real_samples

    async def close(self) -> None:
        pass


class StalledClientWebSocket:
    """Plays scripted frames, yielding between them like a socket read, and holds
    every server frame until it reads."""

    def __init__(self, frames: list[str]) -> None:
        self.frames = frames
        self.sent_events: list[JsonObject] = []
        self.is_reading = asyncio.Event()

    async def receive(self) -> dict[str, str]:
        await asyncio.sleep(0)
        if self.frames:
            return {"type": "websocket.receive", "text": self.frames.pop(0)}
        else:
            return {"type": "websocket.disconnect"}

    async def send_text(self, text: str) -> None:
        await self.is_reading.wait()
        self.sent_events.append(json.loads(text))

    async def close(self) -> None:
        pass


async def open_runtime(adapter: GatedAdapter) -> SessionRuntime:
    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), lambda: adapter, RuntimeLimits()
    )
    await runtime.update({}, "client_update")
    return runtime


async def receive_until(
    runtime: SessionRuntime, event_type: type[Drained | UnitCompleted | Closed]
) -> list[Envelope]:
    envelopes: list[Envelope] = []
    async for envelope in runtime.outputs():
        envelopes.append(envelope)
        if isinstance(envelope.event, event_type):
            break
        else:
            pass
    return envelopes


@pytest.mark.asyncio
async def test_eos_marks_only_the_last_unit_when_end_arrives_mid_backlog() -> None:
    adapter = GatedAdapter([])
    runtime = await open_runtime(adapter)
    await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
    await adapter.has_started.wait()
    await runtime.append(b"\1" * UNIT_BYTES * 2, 1, None, "client_append_1")
    await runtime.end("client_end")

    adapter.release.set()
    drained = (await receive_until(runtime, Drained))[-1].event
    await runtime.close("client_closed")

    assert [unit.eos for unit in adapter.units] == [False, False, True]
    assert isinstance(drained, Drained)
    assert (drained.accepted_end_ms, drained.consumed_ms) == (60.0, 60.0)


@pytest.mark.asyncio
async def test_close_finishes_only_responses_the_client_has_seen() -> None:
    adapter = GatedAdapter([ResponseStarted("seen"), ResponseStarted("unseen")])
    runtime = await open_runtime(adapter)
    await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
    await adapter.has_started.wait()
    adapter.release.set()
    for envelope in await receive_until(runtime, UnitCompleted):
        if envelope.event == ResponseStarted("seen"):
            runtime.output_buffer.before_send(envelope)
            runtime.output_buffer.sent(envelope)
        else:
            pass

    await runtime.close("client_closed")
    closing = [envelope.event for envelope in await receive_until(runtime, Closed)]

    finished = [event for event in closing if isinstance(event, ResponseFinished)]
    assert [event.response_id for event in finished] == ["seen"]
    assert finished[0].status == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "code", "has_traceback"),
    [
        (
            RuntimeError(
                "context_exhausted: the session reached the thinker context length"
            ),
            "context_exhausted",
            False,
        ),
        (RuntimeError("talker step failed"), "internal", True),
        (
            OutputBudgetError("outbound event budget exhausted"),
            "output_budget_exhausted",
            False,
        ),
        (ContextLimitError("response text context limit"), "context_limit", False),
    ],
)
async def test_unit_failure_closes_session_and_logs_once(
    error: Exception,
    code: str,
    has_traceback: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingAdapter(GatedAdapter):
        async def process(self, unit: Unit) -> int:
            raise error

    with caplog.at_level(logging.WARNING, logger=RUNTIME_LOGGER_NAME):
        runtime = await open_runtime(FailingAdapter([]))
        await runtime.append(b"\1" * UNIT_BYTES, 0, None, "append")
        envelopes = await asyncio.wait_for(receive_until(runtime, Closed), 5)
    failures = [entry.event for entry in envelopes if isinstance(entry.event, Failure)]
    assert len(failures) == 1
    assert failures[0].code == code
    assert failures[0].is_fatal
    assert failures[0].message == str(error)
    runtime_records = [
        record for record in caplog.records if record.name == RUNTIME_LOGGER_NAME
    ]
    assert len(runtime_records) == 1
    assert runtime_records[0].levelno == logging.ERROR
    assert (runtime_records[0].exc_info is not None) == has_traceback


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_type", "client_reads", "fatal_codes"),
    [
        ("input_audio_buffer.append", False, ["output_budget_exhausted"]),
        ("unsupported", False, ["output_budget_exhausted"]),
        ("input_audio_buffer.append", True, []),
    ],
)
async def test_client_that_stops_reading_fails_with_the_output_budget_code(
    event_type: str, client_reads: bool, fatal_codes: list[str]
) -> None:
    runtime = SessionRuntime(
        MODEL_NAME,
        Capabilities(),
        lambda: GatedAdapter([]),
        RuntimeLimits(max_output_events=3),
    )
    update = {"type": "session.update", "event_id": "update", "session": {}}
    frames = [
        {
            "type": event_type,
            "event_id": f"frame_{sequence}",
            "audio": base64.b64encode(b"\1\0").decode("ascii"),
            "sglang": {"seq": sequence},
        }
        for sequence in range(MAX_STALLED_FRAMES)
    ]
    websocket = StalledClientWebSocket(
        [json.dumps(frame) for frame in [update, *frames]]
    )
    if client_reads:
        websocket.is_reading.set()
    else:
        pass
    run_task = asyncio.create_task(SharedRealtimeSession(websocket, runtime).run())
    while runtime.close_task is None:
        await asyncio.sleep(0)
    websocket.is_reading.set()
    await asyncio.wait_for(run_task, 5)

    assert [
        event["error"]["code"]
        for event in websocket.sent_events
        if event["type"] == "error" and event["sglang"]["fatal"]
    ] == fatal_codes


def test_output_budget_counts_outbound_events_only() -> None:
    buffer = OutputBuffer(RuntimeLimits(max_output_bytes=1024, max_output_events=2))
    unit = Unit(0, 0, bytes(32000), 16000, images=(bytes(512 * 1024),) * 4)
    completed = Envelope(event=UnitCompleted(unit.unit_id), unit=unit)
    buffer.enqueue(completed)
    with pytest.raises(OutputBudgetError, match="outbound event budget exhausted"):
        buffer.enqueue(Envelope(event=AudioDelta("response", "item", bytes(2048))))
    buffer.enqueue(completed)
    with pytest.raises(OutputBudgetError, match="outbound event budget exhausted"):
        buffer.enqueue(completed)
    assert [buffer.dequeue(), buffer.dequeue(), buffer.dequeue()] == [
        completed,
        completed,
        None,
    ]
    buffer.start_response("first", ("text",))
    buffer.start_response("second", ("text",))
    with pytest.raises(OutputBudgetError, match="unfinished response budget"):
        buffer.start_response("third", ("text",))
