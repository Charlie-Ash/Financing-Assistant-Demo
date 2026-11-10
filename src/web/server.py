# The web interface: one FastAPI app that serves both the chat page
# (static/) and the JSON API it calls, on a single port. Run from inside src/,
# after the RAG and Credit Lookup agents are up:
#
#   python -m web.server            # -> http://127.0.0.1:8080
#
# Same convention as main.py and the agents' a2a_server.py: run as a module
# from src/ so the project's absolute imports resolve.
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from web.config import STATIC_DIR, SUMMARY_DIR, WEB_HOST, WEB_PORT
from web.session import SessionError, SessionManager

# What memory_manager.finalize_and_save names a saved summary: a sanitised
# slug + "_chat_points" (+ "_N" when the name was taken) + ".md". Only such
# files, directly inside SUMMARY_DIR, can be downloaded.
SUMMARY_FILE_RE = re.compile(r"^[a-z0-9_-]+_chat_points(_\d+)?\.md$")


class StartRequest(BaseModel):

    user_id: str = Field(min_length=1, max_length=32, pattern=r"^\S+$")
    role: Literal["sales", "credit"]


class ChatRequest(BaseModel):

    message: str = Field(min_length=1, max_length=2000)


class EndRequest(BaseModel):

    save: bool


def create_app(manager: SessionManager | None = None, summary_dir: Path = SUMMARY_DIR) -> FastAPI:

    manager = manager or SessionManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await manager.close()

    app = FastAPI(title="Financing Assistant", lifespan=lifespan)

    # The page and its JS/CSS must never be reused from the browser cache
    # without asking: after a `git pull` an old app.js would show stale agent
    # labels and help. "no-cache" still allows a cheap 304 when unchanged.
    @app.middleware("http")
    async def revalidate_page_files(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.exception_handler(SessionError)
    async def session_error(request: Request, exc: SessionError):
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.get("/api/session")
    async def get_session():
        return manager.snapshot()

    @app.post("/api/session")
    async def start_session(body: StartRequest):
        return await manager.start(body.user_id, body.role)

    @app.post("/api/chat")
    async def chat(body: ChatRequest):
        message = body.message.strip()
        if not message:
            raise HTTPException(status_code=422, detail="訊息不能是空白。")
        return await manager.chat(message)

    @app.post("/api/session/end")
    async def end_session(body: EndRequest):
        saved_path = await manager.end(body.save)
        if saved_path is None:
            return {"saved": False, "download_url": None}
        return {"saved": True, "file_name": saved_path.name, "download_url": f"/api/summaries/{saved_path.name}"}

    @app.get("/api/summaries/{file_name}")
    async def download_summary(file_name: str):
        path = summary_dir / file_name
        if not SUMMARY_FILE_RE.match(file_name) or not path.is_file():
            raise HTTPException(status_code=404, detail="找不到這份對話摘要。")
        return FileResponse(path, media_type="text/markdown; charset=utf-8", filename=file_name)

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()


def main():

    print(f"Web interface: http://{WEB_HOST}:{WEB_PORT}")
    uvicorn.run(app, host=WEB_HOST, port=WEB_PORT)


if __name__ == "__main__":
    main()
