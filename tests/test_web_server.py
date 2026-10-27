# The web interface's API (src/web/server.py) driven through FastAPI's
# TestClient, with a fake orchestrator in place of the real one -- no agents,
# LLM or network needed. Covers the chat lifecycle the page relies on:
# start, chat, reload, end with/without saving, and the summary download.
import pytest
from fastapi.testclient import TestClient

from web.server import create_app
from web.session import SessionManager


class FakeOrchestrator:

    def __init__(self, caller, summary_dir, fail_with=None):
        self.caller = caller
        self.summary_dir = summary_dir
        self.fail_with = fail_with
        self.closed = False
        self.ended_with = None

    async def run_turn(self, message):
        if self.fail_with:
            raise self.fail_with
        tool = "credit_lookup" if "逾期" in message else "rag"
        return {"reply": f"回覆：{message}", "tool": tool}

    def connected_agents(self):
        return ["rag", "credit_lookup"]

    def end_session(self, save):
        self.ended_with = save
        if not save:
            return None
        path = self.summary_dir / "wang_overdue_check_chat_points.md"
        path.write_text("Summary to question 1: \n王大明無逾期紀錄。\n", encoding="utf-8")
        return path

    async def aclose(self):
        self.closed = True


@pytest.fixture
def harness(tmp_path):

    created = []
    thread_clears = []

    async def factory(caller):
        orchestrator = FakeOrchestrator(caller, tmp_path)
        created.append(orchestrator)
        return orchestrator

    manager = SessionManager(orchestrator_factory=factory, clear_thread_id=lambda: thread_clears.append(True))
    with TestClient(create_app(manager, summary_dir=tmp_path)) as client:
        yield client, created, thread_clears


def _start(client, user_id="wang01", role="sales"):
    return client.post("/api/session", json={"user_id": user_id, "role": role})


def test_page_is_served(harness):

    client, _, _ = harness
    response = client.get("/")

    assert response.status_code == 200
    assert "融資業務助理" in response.text
    assert client.get("/static/app.js").status_code == 200


def test_no_session_until_started(harness):

    client, _, _ = harness

    assert client.get("/api/session").json() == {"active": False}
    assert client.post("/api/chat", json={"message": "你好"}).status_code == 409


def test_start_validates_role_and_user_id(harness):

    client, created, _ = harness

    assert _start(client, role="admin").status_code == 422
    assert _start(client, user_id="").status_code == 422
    assert _start(client, user_id="has space").status_code == 422
    assert created == []


def test_start_creates_a_fresh_chat_with_the_callers_identity(harness):

    client, created, thread_clears = harness
    body = _start(client, role="credit").json()

    assert body["active"] is True
    assert body["caller"] == {"user_id": "wang01", "role": "credit"}
    assert body["role_label"] == "授信"
    assert body["connected_agents"] == ["rag", "credit_lookup"]
    assert created[0].caller == {"user_id": "wang01", "role": "credit"}
    assert thread_clears == [True]  # never silently resumes an old thread


def test_only_one_chat_at_a_time(harness):

    client, _, _ = harness
    _start(client)

    assert _start(client, user_id="lin02").status_code == 409


def test_chat_reports_the_agent_and_time_and_is_kept_for_reload(harness):

    client, _, _ = harness
    _start(client)

    entry = client.post("/api/chat", json={"message": "王大明有沒有逾期？"}).json()

    assert entry["role"] == "assistant"
    assert entry["text"] == "回覆：王大明有沒有逾期？"
    assert entry["agent"] == "credit_lookup"
    assert isinstance(entry["elapsed_s"], float)

    transcript = client.get("/api/session").json()["transcript"]
    assert [m["role"] for m in transcript] == ["user", "assistant"]
    assert transcript[0]["text"] == "王大明有沒有逾期？"


def test_blank_message_is_rejected(harness):

    client, _, _ = harness
    _start(client)

    assert client.post("/api/chat", json={"message": "   "}).status_code == 422


def test_a_failed_turn_becomes_an_error_message_not_a_crash(tmp_path):

    async def factory(caller):
        return FakeOrchestrator(caller, tmp_path, fail_with=RuntimeError("Gemini rate limit (HTTP 429)"))

    with TestClient(create_app(SessionManager(factory, clear_thread_id=lambda: None), summary_dir=tmp_path)) as client:
        _start(client)
        entry = client.post("/api/chat", json={"message": "你好"}).json()

    assert entry["role"] == "error"
    assert "429" in entry["text"]


def test_end_with_save_returns_a_working_download_link(harness):

    client, created, _ = harness
    _start(client)
    client.post("/api/chat", json={"message": "王大明有沒有逾期？"})

    result = client.post("/api/session/end", json={"save": True}).json()

    assert result["saved"] is True
    assert result["download_url"] == "/api/summaries/wang_overdue_check_chat_points.md"
    download = client.get(result["download_url"])
    assert download.status_code == 200
    assert "王大明無逾期紀錄" in download.text
    assert "attachment" in download.headers["content-disposition"]

    assert created[0].ended_with is True and created[0].closed is True
    assert client.get("/api/session").json() == {"active": False}


def test_end_without_save_discards_and_allows_a_new_chat(harness):

    client, created, _ = harness
    _start(client)

    result = client.post("/api/session/end", json={"save": False}).json()

    assert result == {"saved": False, "download_url": None}
    assert created[0].ended_with is False and created[0].closed is True
    assert _start(client, user_id="lin02").status_code == 200


def test_summary_download_only_serves_summary_files(harness, tmp_path):

    client, _, _ = harness
    (tmp_path / "notes.md").write_text("secret", encoding="utf-8")

    assert client.get("/api/summaries/notes.md").status_code == 404
    assert client.get("/api/summaries/..%2Fcredit_demo.db").status_code == 404
    assert client.get("/api/summaries/missing_chat_points.md").status_code == 404
