from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import os
import uuid


from webagent import run_web_agent, get_session_history

app_root_path = os.getenv("ROOT_PATH", "/web")

app = FastAPI(root_path=app_root_path)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

class ChatRequest(BaseModel):
    message: str
    session_id: str
    language: str = "Default English"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, session_id: str = None):
    if not session_id:
        session_id = str(uuid.uuid4())
    return templates.TemplateResponse("index.html", {"request": request, "session_id": session_id})

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    user_query = chat_request.message
    session_id = chat_request.session_id
    language = chat_request.language
    
    response = run_web_agent(user_query, session_id, language)
    
    return {"response": response}

@app.post("/history")
async def history_endpoint(chat_request: ChatRequest):
    session_id = chat_request.session_id
    history = get_session_history(session_id)
    return {"history": history}

class StripPrefixMiddleware:
    def __init__(self, app, prefix):
        self.app = app
        self.prefix = prefix
        
    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path", "")
            if path.startswith(self.prefix):
                scope["path"] = path[len(self.prefix):]
                if not scope["path"]:
                    scope["path"] = "/"
                scope["root_path"] = self.prefix
        return await self.app(scope, receive, send)

if app_root_path and app_root_path != "/":
    app = StripPrefixMiddleware(app, app_root_path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
