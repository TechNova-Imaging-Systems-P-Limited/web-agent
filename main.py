from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import os
import uuid


from webagent import run_web_agent, get_session_history

app = FastAPI()

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
