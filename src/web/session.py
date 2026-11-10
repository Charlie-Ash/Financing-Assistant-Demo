# The web interface's one active chat. Mirrors main.py's REPL: one
# Orchestrator per chat, created with the caller's identity, ended with the
# same save/discard choice as typing "bye". The orchestrator's end-of-chat
# memory is a single shared file (orchestrator/memory_manager.py), so only
# one chat can be active at a time -- multi-user sessions are a later step.
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Optional

from orchestrator import memory_manager

ROLE_LABELS = {"sales": "業務", "credit": "授信"}


class SessionError(Exception):

    """A request that doesn't fit the current state (e.g. chatting with no
    active chat); status_code is the HTTP status the API should answer with."""

    def __init__(self, message: str, status_code: int = 409):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ChatSession:

    orchestrator: object
    caller: dict
    started_at: str
    connected_agents: list[str]
    # Every message shown on the page, so a reload can redraw the chat:
    # {"role": "user" | "assistant" | "error", "text", "agent"?, "elapsed_s"?}
    transcript: list[dict] = field(default_factory=list)


async def _create_orchestrator(caller: dict):

    # Imported here so the web server (and its tests) don't pull in the LLM
    # client and A2A stack until a chat actually starts.
    from orchestrator.orchestrator import Orchestrator

    return await Orchestrator.create(caller)


class SessionManager:

    def __init__(
        self,
        orchestrator_factory: Callable[[dict], Awaitable[object]] = _create_orchestrator,
        clear_thread_id: Callable[[], None] = memory_manager.clear_thread_id,
    ):

        self._factory = orchestrator_factory
        self._clear_thread_id = clear_thread_id
        self._lock = asyncio.Lock()
        self.session: Optional[ChatSession] = None

    def snapshot(self) -> dict:

        if self.session is None:
            return {"active": False}
        return {
            "active": True,
            "caller": self.session.caller,
            "role_label": ROLE_LABELS[self.session.caller["role"]],
            "started_at": self.session.started_at,
            "connected_agents": self.session.connected_agents,
            "transcript": self.session.transcript,
        }

    async def start(self, user_id: str, role: str) -> dict:

        async with self._lock:
            if self.session is not None:
                raise SessionError("已有進行中的對話，請先結束目前的對話。")

            # Every web chat starts a fresh conversation. The CLI resumes a
            # thread left over from a crash, but on a web page that history
            # would be invisible, so it's dropped instead.
            self._clear_thread_id()

            caller = {"user_id": user_id, "role": role}
            orchestrator = await self._factory(caller)
            self.session = ChatSession(
                orchestrator=orchestrator,
                caller=caller,
                started_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                connected_agents=orchestrator.connected_agents(),
            )
            return self.snapshot()

    async def chat(self, message: str) -> dict:

        async with self._lock:
            if self.session is None:
                raise SessionError("目前沒有進行中的對話，請先開始對話。")

            self.session.transcript.append({"role": "user", "text": message})
            start = time.perf_counter()
            try:
                turn = await self.session.orchestrator.run_turn(message)
            except Exception as e:
                # An LLM quota/overload error or an unreachable agent ends up
                # here; show it in the chat instead of failing the page.
                entry = {"role": "error", "text": f"處理這則訊息時發生錯誤：{e}"}
            else:
                entry = {
                    "role": "assistant",
                    "text": turn["reply"],
                    "agent": turn["tool"],
                    "elapsed_s": round(time.perf_counter() - start, 1),
                }
            self.session.transcript.append(entry)
            return entry

    async def end(self, save: bool) -> Optional[Path]:

        async with self._lock:
            if self.session is None:
                raise SessionError("目前沒有進行中的對話。")

            orchestrator = self.session.orchestrator
            try:
                # end_session is synchronous and, when saving, makes one LLM
                # call to name the file -- run it off the event loop.
                saved_path = await asyncio.to_thread(orchestrator.end_session, save)
            finally:
                await orchestrator.aclose()
                self.session = None
            return saved_path

    async def close(self) -> None:

        """Server shutdown: release the orchestrator's connections without
        ending the chat (nothing is saved or discarded)."""

        async with self._lock:
            if self.session is not None:
                await self.session.orchestrator.aclose()
                self.session = None
